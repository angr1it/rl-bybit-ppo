
from __future__ import annotations
import argparse
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig, IQL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    args = ap.parse_args()

    data = np.load(args.dataset_path)
    ds = MDPDataset(
        observations=data["observations"],
        actions=data["actions"].reshape(-1,1),
        rewards=data["rewards"],
        terminals=data["terminals"]
    )

    cfg = IQLConfig(actor_learning_rate=3e-4, critic_learning_rate=3e-4, n_epochs=5, batch_size=256)
    algo = IQL(cfg)
    algo.fit(ds, n_epochs=5, with_timestamp=False, show_progress=True)
    algo.save_model("models/iql_policy.d3")

if __name__ == "__main__":
    main()
