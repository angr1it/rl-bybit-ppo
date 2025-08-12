
from __future__ import annotations
import argparse
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--wandb", type=str, default="false")
    args = ap.parse_args()

    data = np.load(args.dataset_path)
    ds = MDPDataset(
        observations=data["observations"],
        actions=data["actions"].reshape(-1,1),  # discrete actions as labels
        rewards=data["rewards"],
        terminals=data["terminals"]
    )

    # Маленькая DecisionTransformer-конфигурация
    from d3rlpy.algos import DecisionTransformerConfig, DecisionTransformer
    cfg = DecisionTransformerConfig(
        context_size=32,  # длина окна
        encoder_factory=d3rlpy.models.encoders.TransformerEncoderFactory(n_heads=2, n_layers=2, d_model=64),
        learning_rate=3e-4,
        warmup_steps=100,
        n_epochs=5,
        batch_size=64
    )
    algo = DecisionTransformer(cfg)
    algo.fit(ds, n_epochs=5, with_timestamp=False, show_progress=True)
    algo.save_model("models/dt_policy.d3")

if __name__ == "__main__":
    main()
