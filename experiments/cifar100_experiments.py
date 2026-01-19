import os
import time
import math
import pickle
import argparse
from functools import partial
from collections import defaultdict
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import train_state, common_utils
import optax
import optax.contrib  # Explicit import for SAM
import tensorflow as tf
import tensorflow_datasets as tfds

# for hpc easier importing
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.models import ResNet18

# --- hardware setup ---
tf.config.set_visible_devices([], "GPU")
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# configuration & constants
# -----------------------------------------------------------------------------
@dataclass
class TrainingArgs:
    """
    Configuration for training hyperparameters and settings.

    Attributes:
        seed: Random seed for reproducibility.
        n_clean_epochs: Number of clean epochs to run before noise injection blocks.
        n_noisy_epochs: Number of noisy epochs to run during injection blocks.
        epochs: Total number of training epochs.
        lr: Base learning rate.
        schedule: Learning rate schedule type ('onecycle', 'cosine', 'flat').
        optimizer: Optimizer type ('Adam', 'SGD', 'SD', 'SAM').
        rho: Neighborhood size for SAM optimizer (default 0.05).
        noise_level: General noise level parameter (unused in main loop currently).
        shape_a: Hyperparameter placeholder (a).
        freq_b: Hyperparameter placeholder (b).
        scale_c: Probability/Scale for noise injection (sp_prob).
        lr_inc_p: Learning rate increment parameter (p).
        grad_inc_f: Gradient norm scaling factor target.
        decay: If True, uses a decaying noise schedule.
        rise: If True, uses a rising noise schedule.
        use_schedule: If False, disables noise schedule entirely (noise=0).
        use_baseline_norms: If True, loads baseline norms from file for scaling.
        batch_size: Training batch size.
        num_classes: Number of classification classes.
        weight_decay: Weight decay factor for optimizer.
        data_dir: Directory for TFDS data.
        save_dir: Directory to save experiment history.
        baseline_grad_path: Path to pre-computed baseline gradient norms.
    """

    seed: int = 42
    n_clean_epochs: int = 5
    n_noisy_epochs: int = 1
    epochs: int = 100
    lr: float = 0.02
    schedule: str = "onecycle"
    optimizer: str = "Adam"
    rho: float = 0.05
    noise_level: float = 0.0
    shape_a: float = 0.0
    freq_b: float = 0.0
    scale_c: float = 0.0
    lr_inc_p: float = 0.0
    grad_inc_f: float = 1.0
    decay: bool = False
    rise: bool = False
    use_schedule: bool = True
    use_baseline_norms: bool = False

    # fixed constants
    batch_size: int = 256
    num_classes: int = 100
    weight_decay: float = 5e-3
    # specific to user environment
    data_dir: str = "/scratch/gpfs/MELCHIOR/ms0821/tfds_data/"
    save_dir: str = (
        "/scratch/gpfs/MELCHIOR/ms0821/lode_optimisers/production-code/lode-optimizer-v2/curriculum_learning/gradnorm-experiments/force-smooth/"
    )
    baseline_grad_path: str = "baseline_gradnorm_lr.npy"


@dataclass
class NoiseCfg:
    """
    Configuration for image noise injection.

    Attributes:
        img_sigma: Standard deviation for Gaussian noise in [0,1] pixel space.
        sp_prob: Probability for Salt & Pepper noise.
    """

    img_sigma: float = 0.0
    sp_prob: float = 0.0


# -----------------------------------------------------------------------------
# data processing pipeline
# -----------------------------------------------------------------------------
def augment_image(img: tf.Tensor) -> tf.Tensor:
    """
    Applies random data augmentation to a single image.

    Args:
        img: Input image tensor.

    Returns:
        Augmented image tensor.
    """
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return img


def apply_noise(img: tf.Tensor, cfg: NoiseCfg) -> tf.Tensor:
    """
    Applies Gaussian and/or Salt & Pepper noise to an image based on config.

    Args:
        img: Input image tensor (float32, [0,1]).
        cfg: Noise configuration object.

    Returns:
        Noisy image tensor.
    """
    # Gaussian
    if cfg.img_sigma > 0.0:
        noise = tf.random.normal(tf.shape(img), stddev=cfg.img_sigma, dtype=img.dtype)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)

    # Salt & Pepper
    if cfg.sp_prob > 0.0:
        prob = cfg.sp_prob
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        m = tf.random.uniform((h, w, 1), dtype=img.dtype)
        salt = tf.cast(m < prob / 2.0, img.dtype)
        pepper = tf.cast(m > (1.0 - prob / 2.0), img.dtype)
        img = img * (1.0 - salt) * (1.0 - pepper) + salt
    return img


def process_sample(
    x: Dict[str, Any], is_training: bool, cfg: NoiseCfg
) -> Dict[str, Any]:
    """
    Processes a single dataset sample: augments, converts type, adds noise.

    Args:
        x: Dictionary containing 'image' and 'label'.
        is_training: Boolean flag for training mode (enables augmentation).
        cfg: Noise configuration.

    Returns:
        Processed dictionary with float32 image and int32 label.
    """
    image = x["image"]
    if is_training:
        image = augment_image(image)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
    image = apply_noise(image, cfg)
    label = tf.cast(x["label"], tf.int32)
    return {"image": image, "label": label}


def create_dataset(
    dataset_builder,
    split: str,
    batch_size: int,
    is_training: bool,
    cfg: NoiseCfg,
    seed: int = 0,
):
    """
    Creates a JAX-ready data iterator from TensorFlow Datasets.

    Args:
        dataset_builder: TFDS builder object.
        split: Dataset split string (e.g., 'train[:80%]').
        batch_size: Batch size.
        is_training: Boolean indicating training mode.
        cfg: Noise configuration.
        seed: Random seed for shuffling.

    Returns:
        An iterator yielding batches of numpy arrays.
    """
    ds = dataset_builder.as_dataset(split=split)

    if is_training:
        ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=True)

    process_fn = partial(process_sample, is_training=is_training, cfg=cfg)
    ds = ds.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return map(lambda x: jax.tree_util.tree_map(lambda t: t._numpy(), x), ds)


# -----------------------------------------------------------------------------
# model & training state
# -----------------------------------------------------------------------------
class CustomTrainState(train_state.TrainState):
    """
    Custom Flax TrainState that includes batch statistics and current learning rate.
    """

    batch_stats: Any = None
    lr: jnp.array = field(default_factory=lambda: jnp.array(0.0, dtype=jnp.float32))


def create_optimizer(
    name: str, learning_rate: float, weight_decay: float, rho: float = 0.05
) -> optax.GradientTransformation:
    """
    Factory function to create the requested optimizer.

    Args:
        name: Name of the optimizer ('Adam', 'SGD', 'SD', 'SAM').
        learning_rate: Base learning rate.
        weight_decay: Weight decay factor.
        rho: Neighborhood size (only used for SAM).

    Returns:
        An optax GradientTransformation.
    """
    if name == "Adam":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    elif name == "SGD":
        # We chain add_decayed_weights to implement L2 regularization/weight decay for SGD
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        )

    elif name == "SD":
        # Implementation: Weight Decay -> Momentum Trace -> Sign -> Scale(-LR)
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.trace(decay=0.9, nesterov=False),
            optax.stateless(
                lambda updates, params: jax.tree_util.tree_map(jnp.sign, updates)
            ),
            optax.scale(-learning_rate),
        )

    elif name == "SAM":
        # This uses the drop-in SAM wrapper from optax.contrib.
        # sync_period=2 means: Step 1 = Adversarial Perturbation, Step 2 = Weight Update

        # Base Optimizer: SGD with momentum and weight decay
        base_opt = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        )

        # Adversarial Optimizer: Normalized SGD (just moves weights by rho in grad direction)
        adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(rho))

        return optax.contrib.sam(base_opt, adv_opt, sync_period=2)

    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_train_state(rng, learning_rate, total_steps, args: TrainingArgs):
    """
    Initializes the model, parameters, and optimizer state.

    Args:
        rng: JAX PRNGKey.
        learning_rate: Initial learning rate.
        total_steps: Total training steps.
        args: Training arguments.

    Returns:
        Initialized CustomTrainState.
    """
    model = ResNet18(
        args.num_classes,
        channel_list=[64, 128, 256, 512],
        num_blocks_list=[2, 2, 2, 2],
        strides=[1, 1, 2, 2, 2],
        head_p_drop=0.3,
    )

    # Initialize Parameters
    input_shape = (1, 32, 32, 3)
    variables = model.init(rng, jnp.ones(input_shape, jnp.float32), train=False)

    # Create Optimizer
    tx = create_optimizer(
        args.optimizer, learning_rate=1.0, weight_decay=args.weight_decay, rho=args.rho
    )

    return CustomTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
        lr=jnp.array(learning_rate, dtype=jnp.float32),
    )


def get_schedule_fn(schedule_type, init_lr, total_steps):
    """Returns a learning rate schedule function."""
    if schedule_type == "cosine":
        return optax.cosine_decay_schedule(init_lr, total_steps)
    elif schedule_type == "onecycle":
        return optax.cosine_onecycle_schedule(
            transition_steps=total_steps, peak_value=init_lr
        )
    elif schedule_type == "flat":
        return optax.constant_schedule(init_lr)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


# -----------------------------------------------------------------------------
# logging helpers
# -----------------------------------------------------------------------------
def metrics_summary(metrics):
    """Aggregates a list of metric dictionaries into a summary mean."""
    metrics = jax.device_get(metrics)
    metrics = jax.tree_util.tree_map(lambda *args: np.stack(args), *metrics)
    summary = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    return summary


def log_metrics(history, summary, name):
    """Updates history dictionary and prints metrics."""
    print(f"{name}: ", end="", flush=True)
    for key, val in summary.items():
        history[f"{name}_{key}"].append(val)
        print(f"{key} {val:.3f} ", end="")


# -----------------------------------------------------------------------------
# step functions
# -----------------------------------------------------------------------------
def compute_metrics(logits, labels, num_classes=100):
    """Computes cross-entropy loss and accuracy."""
    one_hot = common_utils.onehot(labels, num_classes=num_classes)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


@jax.jit
def train_step(
    state, batch, dropout_rng, base_lr, prev_norm, scale_mask, increase_factor
):
    """
    Executes a single training step with gradient scaling and custom LR adaptation.

    Args:
        state: Current CustomTrainState.
        batch: Data batch.
        dropout_rng: Random key for dropout.
        base_lr: Scheduled learning rate for this step.
        prev_norm: Baseline gradient norm from previous epochs/file.
        scale_mask: 0.0 or 1.0, determines if we use adaptive scaling.
        increase_factor: Factor to scale target gradient norm.

    Returns:
        Updated state and metrics.
    """
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch["image"],
            train=True,
            rngs={"dropout": dropout_rng},
            mutable="batch_stats",
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits, common_utils.onehot(batch["label"], 100)
            )
        )
        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, logits)), grads = grad_fn(state.params)
    metrics = compute_metrics(logits, batch["label"])

    # Custom Gradient Scaling Logic
    grad_norm = optax.global_norm(grads)
    scaled_lr = (increase_factor * prev_norm * base_lr) / (grad_norm + 1e-9)
    new_lr = (scale_mask * scaled_lr) + ((1.0 - scale_mask) * base_lr)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    updates = jax.tree_util.tree_map(lambda u: u * new_lr, updates)  # Manually apply LR
    new_params = optax.apply_updates(state.params, updates)

    # store the metrics
    metrics["grad_norm"] = grad_norm
    metrics["scaled_lr"] = new_lr
    metrics["update_mag"] = grad_norm * new_lr

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        batch_stats=new_model_state["batch_stats"],
        lr=new_lr,
    )
    return new_state, metrics


@jax.jit
def eval_step(state, batch):
    """Executes a single evaluation step."""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return compute_metrics(logits, batch["label"], 100)


def run_eval_phase(state, iterator, steps, name, history, newline=True):
    """
    Runs a full evaluation loop for a specific split (val, test, noisy).
    """
    metrics_list = []
    log_every = max(1, int(steps) - 1)

    for step in range(steps):
        batch = next(iterator)
        metrics = eval_step(state, batch)
        metrics_list.append(metrics)

        # Log only on the final step of the loop
        if step > 0 and step % log_every == 0:
            summary = metrics_summary(metrics_list)
            log_metrics(history, summary, name)

            # Handle the formatting difference (Val uses '| ', Test uses newline)
            if newline:
                print()
            else:
                print("| ", end="")


# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
def run_training(args: TrainingArgs):
    """
    Main training loop.

    Args:
        args: TrainingArgs object containing configuration.

    Returns:
        Tuple of (Final TrainState, History dictionary)
    """
    # Setup Data
    dataset_builder = tfds.builder("cifar100", data_dir=args.data_dir)
    dataset_builder.download_and_prepare()

    # Steps calculation
    num_train = dataset_builder.info.splits["train"].num_examples
    num_test = dataset_builder.info.splits["test"].num_examples
    train_steps = math.ceil(num_train * 0.9 / args.batch_size)
    val_steps = math.ceil(num_train * 0.1 / args.batch_size)
    test_steps = math.ceil(num_test * 0.5 / args.batch_size)

    # Fixed Iterators (Val/Test)
    val_iter = create_dataset(
        dataset_builder, "train[90%:]", args.batch_size, False, NoiseCfg()
    )
    test_iter = create_dataset(
        dataset_builder, "test[:50%]", args.batch_size, False, NoiseCfg()
    )
    # Noisy Test Iterator (uses args.scale_c as probability)
    test_iter_noise = create_dataset(
        dataset_builder,
        "test[50%:]",
        args.batch_size,
        False,
        NoiseCfg(sp_prob=args.scale_c),
    )

    # Setup State
    rng = jax.random.PRNGKey(args.seed)
    params_rng, dropout_rng = jax.random.split(rng)

    schedule_fn = get_schedule_fn(args.schedule, args.lr, train_steps * args.epochs)
    state = create_train_state(params_rng, args.lr, train_steps * args.epochs, args)

    # Load Baselines
    if args.use_baseline_norms:
        try:
            baseline_norms = np.array(np.load(args.baseline_grad_path))
        except FileNotFoundError:
            print(f"Warning: {args.baseline_grad_path} not found. Using ones.")
            baseline_norms = np.ones(args.epochs)
    else:
        baseline_norms = np.ones(args.epochs)

    # --- Noise Schedule Setup ---
    # Check if schedule is enabled by the user
    if not args.use_schedule:
        # If schedule is disabled, force noise to 0.0 everywhere
        noise_vec = jnp.zeros(args.epochs)
    else:
        # If schedule is enabled, follow rise/decay/constant logic
        if args.rise:
            noise_vec = jnp.linspace(0.25, 0.90, args.epochs)
        elif args.decay:
            noise_vec = jnp.linspace(1.0, 0.25, args.epochs)
        else:
            noise_vec = jnp.ones(args.epochs) * args.scale_c

    # grad-norm factor schedule
    if args.grad_inc_f == 0.0:
        factor_vec = jnp.linspace(0.3, 1.5, args.epochs)
    elif args.grad_inc_f == -1.0:
        factor_vec = jnp.linspace(1.5, 0.3, args.epochs)
    else:
        factor_vec = jnp.ones(args.epochs) * args.grad_inc_f

    # Tracking
    history = defaultdict(list)

    # noise injection flags
    N_CLEAN = args.n_clean_epochs  # run N_CLEAN clean epochs before next noise
    N_NOISE = args.n_noisy_epochs  # run N_NOISE noisy epochs before next clean
    trigger_noise = False  # initially start clean
    clean_mode = True  # flag for current mode
    noise_epochs = 0  # counts noisy epochs
    clean_epochs = 0  # counts clean epochs
    scale_mask = (
        0.0  # initial scale mask (0: base learning rate, 1: scaled learning rate)
    )

    print(
        f"Starting training for {args.epochs} epochs with optimizer {args.optimizer}..."
    )
    for epoch in range(args.epochs):

        # determine noise injection for this epoch
        if trigger_noise:
            noise_p = noise_vec[epoch]
            # If explicit schedule was disabled, noise_vec is all 0s, so this stays 0.
            if noise_p > 0:
                print(f"\n[ALERT] Injecting Noise (noise level: {noise_p:.2f})...")
            clean_mode = False
            scale_mask = 1.0
            if noise_epochs < N_NOISE - 1:
                noise_epochs += 1
            else:
                trigger_noise = False  # Reset trigger
                noise_epochs = 0
        else:
            noise_p = 0.0
            clean_mode = True
            scale_mask = 0.0
            clean_epochs += 1

        # track the noise level of the epoch
        history["train_noise_level"].append(float(noise_p))

        # prepare train iterator for this epoch
        train_iter = create_dataset(
            dataset_builder,
            "train[:90%]",
            args.batch_size,
            True,
            NoiseCfg(sp_prob=noise_p),
            seed=args.seed + epoch,
        )

        # epoch Loop
        print(f"\n➤  Epoch {epoch+1}/{args.epochs}", end="", flush=True)
        print("\n--- train information ----")
        print(f"Noise: {noise_p:.4f} ", end="", flush=True)
        start_t = time.time()

        epoch_metrics = []
        epoch_norms = []

        for step in range(train_steps):
            batch = next(train_iter)
            # explicitly update the lr so we can adaptively change it inside a JIT
            base_lr = schedule_fn(state.step)

            # Train Step
            state, metrics = train_step(
                state,
                batch,
                dropout_rng,
                base_lr,
                baseline_norms[epoch],
                scale_mask,
                factor_vec[epoch],
            )

            epoch_metrics.append(metrics)
            epoch_norms.append(metrics["grad_norm"])

            if step > 0 and step % (train_steps - 1) == 0:
                summary = {
                    k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]
                }
                history["train_lr"].append(float(base_lr))
                log_metrics(history, summary, "train")
                print(
                    f"| Train Acc: {summary['accuracy']:.3f} Loss: {summary['loss']:.3f} ",
                    end="",
                )

        # if scaling lr by adaptive norm log the most recent 3 update step values
        if not args.use_baseline_norms and clean_mode:
            recent_norm_ave = np.mean(epoch_norms[-3:])
            # update baseline norms to reflect last value
            if epoch + 1 < args.epochs:
                baseline_norms[epoch + 1] = recent_norm_ave

        # record norm values
        avg_norm = np.mean(epoch_norms)
        history["epoch_grad_norm"].append(avg_norm)

        # check for noise injection condition
        if clean_mode and clean_epochs >= N_CLEAN and epoch < 95:
            trigger_noise = True
            clean_epochs = 0

        # Validation & Testing
        run_eval_phase(state, val_iter, val_steps, "val", history, newline=False)
        # Clean Test (prints newline)
        run_eval_phase(state, test_iter, test_steps, "test", history, newline=True)
        # Noisy Test (prints newline)
        run_eval_phase(
            state, test_iter_noise, test_steps, "noisy-test", history, newline=True
        )
        print(f"| time: {time.time() - start_t:.1f}s")

    return state, history


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-100 Training with Noise Injections and Custom Optimizers"
    )

    # Training Config
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Initial learning rate")
    parser.add_argument(
        "--schedule", type=str, default="onecycle", help="LR schedule type"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD", "SD", "SAM"],
        help="Optimizer choice: Adam, SGD, SD (Sign Descent), or SAM",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="Rho parameter for SAM optimizer (neighborhood size)",
    )

    # Noise Injection Config
    parser.add_argument(
        "--n_clean", type=int, default=5, help="Number of clean epochs between noise"
    )
    parser.add_argument(
        "--n_noisy", type=int, default=1, help="Number of noisy epochs per injection"
    )
    parser.add_argument("--noise", type=float, default=0.0, help="Base noise level")
    parser.add_argument(
        "--c", type=float, default=0.0, help="Noise probability/scale (sp_prob)"
    )
    parser.add_argument(
        "--decay", action="store_true", help="Use decaying noise schedule"
    )
    parser.add_argument("--rise", action="store_true", help="Use rising noise schedule")
    parser.add_argument(
        "--use_schedule",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Enable or disable noise schedule entirely",
    )

    # Gradient/Adaptive Config
    parser.add_argument(
        "--a", type=float, default=0.0, help="Shape parameter A (unused)"
    )
    parser.add_argument(
        "--b", type=float, default=0.0, help="Frequency parameter B (unused)"
    )
    parser.add_argument(
        "--p", type=float, default=0.0, help="LR increment param (unused)"
    )
    parser.add_argument(
        "--f", type=float, default=1.0, help="Gradient norm increase factor"
    )
    parser.add_argument(
        "--use_baseline",
        action="store_true",
        help="Use baseline gradient norms from file",
    )

    pargs = parser.parse_args()

    # Parse boolean string for use_schedule
    use_schedule_bool = pargs.use_schedule == "True"

    args = TrainingArgs(
        seed=pargs.seed,
        n_clean_epochs=pargs.n_clean,
        epochs=pargs.epochs,
        n_noisy_epochs=pargs.n_noisy,
        lr=pargs.lr,
        schedule=pargs.schedule,
        optimizer=pargs.optimizer,
        rho=pargs.rho,
        noise_level=pargs.noise,
        shape_a=pargs.a,
        freq_b=pargs.b,
        scale_c=pargs.c,
        lr_inc_p=pargs.p,
        grad_inc_f=pargs.f,
        decay=pargs.decay,
        rise=pargs.rise,
        use_schedule=use_schedule_bool,
        use_baseline_norms=pargs.use_baseline,
    )

    # run
    final_state, history = run_training(args)

    # save directories
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create saved_models folder
    model_dir = os.path.join(args.save_dir, "saved_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    filename_base = (
        f"cifar100_opt_{args.optimizer}_noise_{args.scale_c}_schedule_{args.use_schedule}_"
        f"grad_{args.grad_inc_f}_seed_{args.seed}"
    )

    # Save History
    with open(os.path.join(args.save_dir, f"{filename_base}.pkl"), "wb") as f:
        pickle.dump(history, f)

    # Save Model Parameters (binary)
    model_path = os.path.join(model_dir, f"{filename_base}.params")
    with open(model_path, "wb") as f:
        f.write(serialization.to_bytes(final_state.params))

    print(f"Training complete.")
    print(f"History saved to: {os.path.join(args.save_dir, f'{filename_base}.pkl')}")
    print(f"Model params saved to: {model_path}")
