from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np

def purged_kfold_indices(n: int, n_splits: int=5, embargo_pct: float=0.01) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Генерирует (train_idx, test_idx) для Purged K-Fold с эмбарго.
    n — длина ряда (число наблюдений).
    embargo_pct — доля наблюдений после тестовой выборки, которые исключаются из train.
    """
    fold_sizes = (n // n_splits) * np.ones(n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    current = 0
    indices = np.arange(n)
    embargo = int(np.ceil(n * embargo_pct))

    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((start, stop))
        current = stop

    for i, (start, stop) in enumerate(folds):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[start:stop] = True
        train_mask = ~test_mask

        # Purge: удаляем из train наблюдения, пересекающиеся по времени с тестом (упрощённо: сам фолд)
        train_mask[start:stop] = False
        # Embargo: удаляем часть наблюдений сразу после теста
        embargo_start = stop
        embargo_end = min(n, stop + embargo)
        train_mask[embargo_start:embargo_end] = False

        yield indices[train_mask], indices[test_mask]
