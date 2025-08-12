from __future__ import annotations

import argparse

from .trainer import SB3Trainer, TrainConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Напр. BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--save_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=None)
    args = ap.parse_args()

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
        # базовая среда SingleAssetDailyEnv по умолчанию (env_class=None)
        policy="MlpPolicy",
        policy_kwargs={},  # можно передать net_arch и т.п.
        use_wandb=False,
    )

    SB3Trainer(cfg).train()

if __name__ == "__main__":
    main()
