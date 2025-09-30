import argparse
import datetime
import glob
import os
import pickle
import tempfile
from functools import partial
from typing import NamedTuple

import cairosvg
import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from network import AZNet
from train import Config, recurrent_fn, forward

# To avoid error on unpickling
from train import SelfplayOutput, Sample


def play_game(model, rng_key):
    """Play a single game of 2048 and return the state history."""
    env = pgx.make(config.env_id)
    state = env.init(rng_key)
    history = [state]
    model_params, model_state = model

    while not state.terminated:
        rng_key, step_key, policy_key = jax.random.split(rng_key, 3)

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=policy_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        action = policy_output.action

        state = env.step(state, action, step_key)
        history.append(state)
        print(f"Step: {len(history)}, Score: {state.rewards.sum()}")

    return history


def create_movie(history, output_path="gameplay.mp4"):
    """Create a video from the game history."""
    print("Generating video...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frames = []
        for i, state in enumerate(history):
            svg_path = os.path.join(tmpdir, f"{i:04d}.svg")
            png_path = os.path.join(tmpdir, f"{i:04d}.png")
            state.save_svg(svg_path)
            cairosvg.svg2png(url=svg_path, write_to=png_path, scale=0.5)
            frames.append(imageio.imread(png_path))
        imageio.mimsave(output_path, frames, fps=2)
    print(f"Video saved to {output_path}")


def find_latest_checkpoint():
    """Find the latest checkpoint file."""
    checkpoint_dirs = glob.glob("checkpoints/*")
    if not checkpoint_dirs:
        return None
    latest_dir = max(checkpoint_dirs)
    checkpoints = glob.glob(os.path.join(latest_dir, "*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to the model checkpoint. If not specified, the latest checkpoint will be used.")
    parser.add_argument("--output_path", type=str, default="gameplay.mp4",
                        help="Path to save the output video.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the game.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train a model first with `train.py`.")
        exit()

    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    config: Config = data["config"]
    model = data["model"]
    rng_key = jax.random.PRNGKey(args.seed)

    # We need to define the forward function again for the loaded model
    forward = hk.without_apply_rng(hk.transform_with_state(
        lambda x, is_eval=False: AZNet(
            num_actions=pgx.make(config.env_id).num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )(x, is_training=not is_eval, test_local_stats=False)
    ))

    history = play_game(model, rng_key)
    create_movie(history, args.output_path)
