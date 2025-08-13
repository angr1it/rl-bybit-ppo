# rl-bybit-ppo-template

A modular framework for training and evaluating reinforcement-learning trading strategies on **Bybit** (daily/weekly intervals).
It is organised into separate layers:

1. **Data & backtesting** – ccxt market access, Parquet/DuckDB storage and vectorbt runners.
2. **RL layer** – Stable-Baselines3 PPO with custom Gymnasium environments.
3. **Validation** – walk-forward / purged k-fold splits and basic metrics.

See [`docs/`](docs/index.md) for detailed documentation of the main components.

## Быстрый старт

```bash
# 1) Локальная установка
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Скопируйте .env
cp .env.example .env  # и впишите BYBIT_API_KEY/BYBIT_API_SECRET если планируете paper/live

# 3) Загрузите данные
python scripts/fetch_ohlcv.py --symbols BTC/USDT ETH/USDT --timeframe 1d --since 2021-01-01 --until 2025-01-01 --sandbox true

# 4) Бейзлайн-бэктест (vectorbt)
python scripts/run_backtest_vectorbt.py --symbol BTC/USDT --timeframe 1d

# 5) Обучение PPO
python -m packages.rl.train_ppo --symbol BTC/USDT --timeframe 1d --window 128 --total_timesteps 200000

# 6) Инференс (генерация сигналов)
python -m packages.rl.inference --symbol BTC/USDT --timeframe 1d --window 128 --model_path models/ppo-BTCUSDT-1d.zip

# 7) Jupyter
jupyter lab
```

## Docker (опционально)
```bash
docker compose up --build
# Откройте браузер: http://localhost:8888  (токен см. в логах)
```

## Структура
```
packages/
  data/        # ccxt-клиент, загрузка OHLCV, сохранение в Parquet/DuckDB
  backtest/    # vectorbt runner, комиссии/метрики
  env/         # Gymnasium-энв для дневной/недельной ребалансировки
  rl/          # обучение PPO (SB3), инференс
  validate/    # разбиения purged k-fold, базовые метрики
notebooks/     # серии ноутбуков 01..06
scripts/       # CLI-скрипты: загрузка данных, бэктест
data/          # локальное хранилище (Parquet)
models/        # чекпойнты PPO
config/        # yaml-конфиги (опц.)
```

## Примечания
- Используется **pip**, без Poetry.
- Базовая биржа — **Bybit** (через ccxt). Для теста включайте `--sandbox true` или переменную окружения `SANDBOX=true`.
- Бэктест — **vectorbt**, для «пошагового» режима можете добавить Backtrader.
- RL — **Stable-Baselines3 / PPO** (MlpPolicy); позже можно перейти к `RecurrentPPO` из sb3-contrib.

---

## Новые возможности

### Модели
- **SB3 + крошечный Transformer** в роли `features_extractor` (`packages/rl/policies/transformer_extractor.py`), обучение: `packages/rl/train_ppo_transformer.py`.
- **SB3-contrib RecurrentPPO (LSTM)** (`packages/rl/train_ppo_lstm.py`).
- **TorchRL + мини-Transformer** (`packages/rl_torchrl/train_torchrl_transformer.py`).

### Offline RL
- **Decision Transformer (d3rlpy)** и **IQL/CQL**: сбор траекторий из исторических данных (`packages/offline/build_trajectories.py`), тренировки в `packages/offline/train_dt.py` и `packages/offline/train_iql.py`.

### Конфиги / Трекинг
- **Hydra** для конфигов (`config/`) и **Weights & Biases** (опционально). Включайте W&B флагом `--wandb true`.

### Доп. энвы
- **Оконный** env для трансформеров: `WindowedSingleAssetEnv` (`packages/env/trading_env_windowed.py`).

## Примеры команд

```bash
# Обучение PPO + Transformer extractor на оконном env
python -m packages.rl.train_ppo_transformer --symbol BTC/USDT --timeframe 1d --window 64 --total_timesteps 100000 --wandb false

# Обучение RecurrentPPO (LSTM) на "плоском" env
python -m packages.rl.train_ppo_lstm --symbol BTC/USDT --timeframe 1d --total_timesteps 100000 --wandb false

# TorchRL-трансформер
python -m packages.rl_torchrl.train_torchrl_transformer --symbol BTC/USDT --timeframe 1d --window 64 --steps 5000

# Offline RL: сбор траекторий из SMA-политики и тренировка Decision Transformer
python -m packages.offline.build_trajectories --symbol BTC/USDT --timeframe 1d --window 64
python -m packages.offline.train_dt --dataset_path data/offline/BTCUSDT-1d-window64.npz --wandb false
```
