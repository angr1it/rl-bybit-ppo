from __future__ import annotations

import argparse

from .trainer import SB3Trainer, TrainConfig
from ..env.trading_env_windowed import WindowedSingleAssetEnv
from .policies.transformer_extractor import TransformerFeatureExtractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--total_timesteps", type=int, default=100_000)
    ap.add_argument("--save_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    # опциональные оверрайды
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=None)
    args = ap.parse_args()

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(d_model=32, nhead=4, num_layers=1, out_dim=64),
        net_arch=[64, 64],
    )

    cfg = TrainConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        window=args.window,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        seed=args.seed,
        tb=args.tb,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env_class=WindowedSingleAssetEnv,
        env_kwargs={"window": args.window},
        use_wandb=args.wandb,
        wandb_run_name=f"ppo-transformer-{args.symbol.replace('/','')}-{args.timeframe}",
    )

    SB3Trainer(cfg).train()

if __name__ == "__main__":
    main()
