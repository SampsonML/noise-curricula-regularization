# -------------------------------------- #
#         ImageNet ResNet50              #
#    with Noise Schedules & Optimizers   #
# -------------------------------------- #
import os
import time
import argparse
import pickle
from functools import partial
from typing import Any, Tuple, Dict
from dataclasses import dataclass

# Set TF to CPU-only to avoid memory conflicts with JAX
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import optax
import optax.contrib
import tensorflow_datasets as tfds

# Import your model (ensure resnet.py is in the python path)
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.models import ResNet50

# JAX Config
os.environ["JAX_COMPILATION_CACHE"] = "/scratch/gpfs/ms0821/jax_cache"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class TrainingArgs:
    seed: int = 42
    epochs: int = 50
    init_lr: float = 0.05
    final_lr: float = 1e-5
    schedule: str = "cosine"
    optimizer: str = "SGD"  # Options: Adam, SGD, SD, SAM
    rho: float = 0.05  # SAM Neighborhood

    # Noise Schedule Params
    n_clean: int = 5
    n_noisy: int = 1
    scale_c: float = 0.0
    rise: bool = False
    decay: bool = False
    use_schedule: bool = True

    # Paths
    data_dir: str = "/scratch/gpfs/ms0821/imagenet_data"
    save_dir: str = "imagenet_experiments"


# -----------------------------------------------------------------------------
# Optimizers
# -----------------------------------------------------------------------------
def create_optimizer(
    name: str, learning_rate, weight_decay: float = 1e-4, rho: float = 0.05
):
    """Factory to create optimizers (Adam, SGD, SD, SAM)."""

    # Base scheduler wrapper if learning_rate is a callable
    # Note: Optax expects scalar or schedule.

    if name == "Adam":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    elif name == "SGD":
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        )

    elif name == 'SD':
        # Implementation: Weight Decay -> Momentum Trace -> Sign -> Scale(-LR)
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.trace(decay=0.9, nesterov=False),
            # Wrap the lambda in optax.stateless to provide the necessary .init() method
            optax.stateless(lambda updates: jax.tree_map(jnp.sign, updates)), 
            optax.scale(-learning_rate)
        )

    elif name == "SAM":
        # SAM with SGD base
        base_opt = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        )
        adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(rho))
        return optax.contrib.sam(base_opt, adv_opt, sync_period=2)

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# -----------------------------------------------------------------------------
# Data Pipeline Helpers
# -----------------------------------------------------------------------------
# We use a TF Variable to control noise level dynamically without rebuilding the graph
NOISE_VAR = tf.Variable(0.0, dtype=tf.float32, trainable=False)


def cutout(image, mask_size=50):
    height, width, _ = image.shape
    cutout_center_h = tf.random.uniform([], 0, height, dtype=tf.int32)
    cutout_center_w = tf.random.uniform([], 0, width, dtype=tf.int32)
    lower_h = tf.maximum(0, cutout_center_h - mask_size // 2)
    upper_h = tf.minimum(height, cutout_center_h + mask_size // 2)
    lower_w = tf.maximum(0, cutout_center_w - mask_size // 2)
    upper_w = tf.minimum(width, cutout_center_w + mask_size // 2)
    mask = tf.ones([upper_h - lower_h, upper_w - lower_w], dtype=tf.float32)
    mask = tf.pad(
        mask,
        [[lower_h, height - upper_h], [lower_w, width - upper_w]],
        constant_values=0,
    )
    mask = tf.expand_dims(mask, axis=-1)
    image = image * (1.0 - mask)
    return image


def apply_noise_tf(image, label):
    """Applies Salt & Pepper noise based on the global NOISE_VAR."""
    # Only apply if probability > 0
    prob = NOISE_VAR

    # Logic: Generate random mask, if val < prob/2 -> Salt (1), if val > 1-prob/2 -> Pepper (0)
    # We use a conditional to avoid expensive ops if prob is 0

    def noisy_img():
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        m = tf.random.uniform((h, w, 1), dtype=image.dtype)
        salt = tf.cast(m < prob / 2.0, image.dtype)
        pepper = tf.cast(m > (1.0 - prob / 2.0), image.dtype)
        return image * (1.0 - salt) * (1.0 - pepper) + salt

    # Optimization: Use tf.cond to skip noise generation if prob is 0
    image = tf.cond(prob > 0.0, noisy_img, lambda: image)
    return image, label


def augment(image, label):
    image = tf.image.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    image = cutout(image, mask_size=50)
    return image, label


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    imagenet_mean = tf.constant([0.485, 0.456, 0.406])
    imagenet_std = tf.constant([0.229, 0.224, 0.225])
    image = (image - imagenet_mean) / imagenet_std
    return image, label


def center_crop(image, label):
    image = tf.image.resize(image, [256, 256])
    image = tf.image.central_crop(image, central_fraction=224 / 256)
    return image, label


# -----------------------------------------------------------------------------
# Training State & Schedule
# -----------------------------------------------------------------------------
def get_schedule(schedule_type, init_lr, total_steps):
    if schedule_type == "cosine":
        return optax.cosine_decay_schedule(init_lr, total_steps)
    elif schedule_type == "onecycle":
        return optax.cosine_onecycle_schedule(
            total_steps, init_lr, pct_start=0.3, div_factor=20.0, final_div_factor=1e4
        )
    elif schedule_type == "flat":
        return optax.constant_schedule(init_lr)
    elif schedule_type == "decay":
        return optax.exponential_decay(
            init_lr, max(total_steps // 3, 1), 0.5, staircase=True
        )
    else:
        raise ValueError(f"Unsupported schedule: {schedule_type}")


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def create_train_state(rng, args, total_steps, model):
    schedule_fn = get_schedule(args.schedule, args.init_lr, total_steps)

    # Use our factory
    optimizer = create_optimizer(
        args.optimizer, schedule_fn, weight_decay=1e-4, rho=args.rho
    )

    variables = model.init({"params": rng}, jnp.ones([1, 224, 224, 3]), train=True)

    return (
        TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optimizer,
        ),
        schedule_fn,
    )


# -----------------------------------------------------------------------------
# Step Functions
# -----------------------------------------------------------------------------
@jax.jit
def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * nn.log_softmax(logits), axis=-1))


@jax.jit
def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return {"loss": loss, "accuracy": accuracy}


def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updated_model_state = state.apply_fn(
            variables, batch["image"], mutable=["batch_stats"], train=True
        )
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, (logits, updated_model_state["batch_stats"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    metrics = compute_metrics(logits, batch["label"])
    return state, metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument(
        "--optimizer", type=str, default="SGD", choices=["Adam", "SGD", "SD", "SAM"]
    )
    parser.add_argument("--rho", type=float, default=0.05)

    # Noise Params
    parser.add_argument("--n_clean", type=int, default=5)
    parser.add_argument("--n_noisy", type=int, default=1)
    parser.add_argument("--scale_c", type=float, default=0.0)  # Noise Probability
    parser.add_argument("--rise", action="store_true")
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--use_schedule", type=str, default="True")

    pargs = parser.parse_args()

    args = TrainingArgs(
        seed=pargs.seed,
        epochs=pargs.epochs,
        init_lr=pargs.init_lr,
        final_lr=pargs.final_lr,
        schedule=pargs.schedule,
        optimizer=pargs.optimizer,
        rho=pargs.rho,
        n_clean=pargs.n_clean,
        n_noisy=pargs.n_noisy,
        scale_c=pargs.scale_c,
        rise=pargs.rise,
        decay=pargs.decay,
        use_schedule=(pargs.use_schedule == "True"),
    )

    # Setup Data Pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 256

    # Load raw
    train_dir = os.path.join(args.data_dir, "ILSVRC/Data/CLS-LOC/train")
    val_dir = os.path.join(args.data_dir, "val_split_new")  # Ensure this exists

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(256, 256), batch_size=None
    )

    # Pipeline: Augment -> Noise -> Normalize -> Batch
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(
        apply_noise_tf, num_parallel_calls=AUTOTUNE
    )  # Uses dynamic NOISE_VAR
    train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(AUTOTUNE)

    # Val/Test Pipeline
    val_ds_full = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(256, 256), batch_size=None
    )
    val_size = int(0.8 * 50000)
    val_ds = val_ds_full.take(val_size)
    test_ds = val_ds_full.skip(val_size)

    def process_eval(ds):
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
        ds = ds.map(center_crop, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    val_ds = process_eval(val_ds)
    test_ds = process_eval(test_ds)

    # Setup Noise Schedule Vectors
    if not args.use_schedule:
        noise_vec = np.zeros(args.epochs)
    elif args.rise:
        noise_vec = np.linspace(0.1, 0.9, args.epochs)
    elif args.decay:
        noise_vec = np.linspace(1.0, 0.1, args.epochs)
    else:
        noise_vec = np.ones(args.epochs) * args.scale_c

    # Setup Model & State
    # Need to approximate steps (ImageNet train is ~1.28M images)
    steps_per_epoch = 1281167 // batch_size
    total_steps = steps_per_epoch * args.epochs
    save_freq = total_steps // 20  # Save 20 times total (or adjust as needed)

    print(f"Start Training: {args.epochs} Epochs, {steps_per_epoch} Steps/Epoch")
    print(f"Optimizer: {args.optimizer}, Noise C: {args.scale_c}")

    model = ResNet50(num_classes=1000)
    rng = jax.random.PRNGKey(args.seed)
    state, lr_fn = create_train_state(rng, args, total_steps, model)

    # Directories
    os.makedirs(args.save_dir, exist_ok=True)
    models_dir = os.path.join(args.save_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    # Tracking
    history = {
        k: []
        for k in ["loss", "lr", "val_loss", "val_acc", "test_loss", "test_acc", "noise"]
    }

    # Noise State
    trigger_noise = False
    clean_cnt = 0
    noise_cnt = 0
    global_step = 0

    for epoch in range(args.epochs):
        # --- Noise Logic ---
        current_noise = 0.0
        if trigger_noise:
            current_noise = noise_vec[epoch]
            if noise_cnt < args.n_noisy - 1:
                noise_cnt += 1
            else:
                trigger_noise = False
                noise_cnt = 0
        else:
            current_noise = 0.0
            clean_cnt += 1
            if clean_cnt >= args.n_clean:
                trigger_noise = True
                clean_cnt = 0

        # Update TF Variable for Data Pipeline
        NOISE_VAR.assign(float(current_noise))
        history["noise"].append(current_noise)

        print(f"\nEpoch {epoch+1}/{args.epochs} | Noise Level: {current_noise:.4f}")

        # --- Training Loop ---
        epoch_loss = []
        start_t = time.time()

        for i, batch in enumerate(train_ds):
            global_step += 1
            # Data is already normalized by TF pipeline, just convert to JAX
            # REMOVE the /255.0 from the original script logic here!
            jax_batch = {
                "image": jnp.array(batch[0]),
                "label": one_hot(jnp.array(batch[1]), 1000),
            }

            state, metrics = train_step(state, jax_batch)
            epoch_loss.append(metrics["loss"])

            if i % 1000 == 0:
                print(
                    f"  Step {i}/{steps_per_epoch} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}"
                )

            # --- Evaluation & Saving ---
            if global_step % save_freq == 0:
                print("  Running Evaluation...")

                def run_eval(ds):
                    losses, accs = [], []
                    count = 0
                    for b in ds:
                        # Limit val set for speed if needed, or run full
                        if count > 50:
                            break
                        logits = state.apply_fn(
                            {"params": state.params, "batch_stats": state.batch_stats},
                            jnp.array(b[0]),
                            train=False,
                            mutable=False,
                        )
                        m = compute_metrics(logits, one_hot(jnp.array(b[1]), 1000))
                        losses.append(m["loss"])
                        accs.append(m["accuracy"])
                        count += 1
                    return np.mean(losses), np.mean(accs)

                v_loss, v_acc = run_eval(val_ds)
                t_loss, t_acc = run_eval(test_ds)

                print(f"  [Eval] Val Acc: {v_acc:.4f} | Test Acc: {t_acc:.4f}")

                history["loss"].append(
                    np.mean(epoch_loss[-100:])
                )  # last 100 steps mean
                history["lr"].append(lr_fn(global_step))
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                history["test_loss"].append(t_loss)
                history["test_acc"].append(t_acc)

                # Save History
                fname = f"resnet50_{args.optimizer}_seed_{args.seed}"
                with open(
                    os.path.join(args.save_dir, f"{fname}_history.pkl"), "wb"
                ) as f:
                    pickle.dump(history, f)

                # Save Model
                with open(
                    os.path.join(models_dir, f"{fname}_step_{global_step}.params"), "wb"
                ) as f:
                    f.write(serialization.to_bytes(state.params))

        print(f"Epoch Time: {time.time() - start_t:.1f}s")


if __name__ == "__main__":
    main()
