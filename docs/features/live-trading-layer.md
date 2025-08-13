# Документация по фиче: Bybit live trading layer

---

## 1. Контекст до внедрения фичи

Репозиторий `rl-bybit-ppo` реализует оффлайн цикл обучения и проверки стратегий на данных Bybit: загрузка котировок → подготовка признаков → обучение PPO → оффлайн-инференс и бэктест. Все вычисления проводятся оффлайн, а агенты взаимодействуют с окружением, где нормировка признаков выполняется внутри среды.

Сейчас отсутствует единая реализация **онлайн‑торговли**. Нет слоя, который позволил бы запускать обученную стратегию в paper или live режимах на Bybit v5. В артефактах обучения не сохраняются статистики нормировки признаков, что мешает воспроизводить результаты вне обучающей среды. Инференс поддерживает только обычный PPO без `RecurrentPPO`.

Ограничения текущей системы:

- Невозможно выполнить live‑торговлю или paper симуляцию, сохраняя консистентность признаков.
- Реализована только оффлайн нормировка в `trading_env`; статистики (`mu`, `sd`) не сохраняются.
- `inference.py` не работает с `RecurrentPPO` и не умеет поддерживать скрытое состояние между шагами.

Затронутые пользователи:

- Разработчики стратегий, желающие довести PPO‑агента до live торговли.
- Трейдеры/исследователи, которым нужен paper режим для тестирования.
- Интеграции и инфраструктура для Bybit API (REST/WS).

---

## 2. Цель фичи

- **Проблема**: отсутствие онлайнового слоя и сохранённых статистик признаков делает невозможным запуск стратегии вне оффлайн среды.
- **Цель**: реализовать полнофункциональный live‑pipeline от потоковых котировок до исполнения ордеров на Bybit v5, с поддержкой paper режима и RecurrentPPO. Обеспечить строгую консистентность признаков через сохранённые статистики и идентичный pipeline фич.
- **Критерии успеха**:
  - Артефакт тренера содержит `feature_stats` (mu, sd, columns) и параметры комиссий.
  - Live runner загружает агента и нормировку из артефакта и выдаёт решения на закрытии бара t для позиции t+1.
  - Поддерживается как обычный PPO, так и `RecurrentPPO` с состоянием и `episode_starts`.
  - Наличие двух режимов брокера: `paper` (симуляция с FeeModel) и `bybit` (REST+WS), оба работающих через единый интерфейс.
  - CLI‑скрипты позволяют запускать paper/live режим с указанной моделью и конфигом.

---

## 3. Решение

### 3.1 Что входит в фичу (Scope)

- Добавление слоя `packages/live/` с модулями:
  - `models.py` – датаклассы для ордеров, позиций, котировок и свечей. Подробнее см. в [data contracts](../live/data-contracts.md).
  - `broker/` – интерфейс `Broker` и реализации `bybit.py` (REST+WS) и `paper.py` (симулятор). См. [broker interface](../live/broker.md).
  - `stream.py` – поток маркет‑данных: WebSocket с fallback на REST (см. [stream and features](../live/stream-features.md)).
  - `features_live.py` – инкрементальный расчёт признаков, 1:1 с `packages/env/features.py`. Детали в [stream and features](../live/stream-features.md).
  - `execution.py` – планирование ордеров из дискретных действий агента.
  - `risk.py` – предторговые лимиты, kill‑switch, circuit breaker. Описание в [execution and risk](../live/execution-risk.md).
  - `runner.py` – главный цикл: поток данных → признаки → агент → риск → исполнение. См. [live runner](../live/runner.md).
- Центральный класс конфигурации `Settings` в `packages/core/settings.py` управляет переменными окружения через `pydantic_settings` и кэшируется с помощью `lru_cache` (см. [settings](../settings.md)).
- Модификация `packages/rl/trainer.py` для сохранения `feature_stats` (mu, sd, columns) и параметров `FeeModel`.
- Обновление `packages/rl/inference.py` с поддержкой `RecurrentPPO`.
- Расширение `packages/data/config.py` настройками live режима и ENV переменными.
- Добавление hydra‑конфигов `config/live/bybit_paper.yaml` и `config/live/bybit_live.yaml`.
- CLI‑скрипты `scripts/live_paper.py` и `scripts/live_trade.py` для запуска runner.
- Документация и шаблон `.env` с ключами Bybit и параметрами риска.

### 3.2 Что не входит (Out-of-scope)

- Поддержка других бирж кроме Bybit v5.
- Реализация UI или панелей мониторинга.
- Полная система алертинга и метрик (оставляется как опциональная часть).

### 3.3 Как это должно работать

1. Тренер сохраняет статистики признаков и параметры комиссий в мета‑артефакт.
2. Live runner загружает модель и метаданные, создаёт `OnlineFeaturePipe` с теми же колонками и нормировкой.
3. Поток `Kline`/`Quote` формирует закрытые бары. На закрытии бара t агент получает нормализованные признаки и выдаёт действие для бара t+1.
4. `ExecutionPlanner` переводит целевое действие {-1,0,+1} в набор ордеров с учётом шага, `reduce_only`, `post_only`, `TIF`.
5. `RiskEngine` проверяет лимиты (макс позиция, дневной убыток, price band, частота ордеров). При нарушении активируется kill‑switch.
6. `Broker` (paper или Bybit) отправляет заявки, обрабатывает репорты и позиции, обеспечивает реконнект и idempotency по `cl_id`.

### 3.4 Архитектурные изменения

- Новая директория `packages/live/` с многочисленными модулями.
- Изменения в `packages/rl/` (trainer, inference) для поддержки новых артефактов.
- Расширение `packages/data/config.py` и новых конфигов `config/live/*`.
- Дополнение `scripts/` новыми CLI для paper/live запусков.
- Интеграция `FeeModel` из `packages/backtest/fees.py` в paper брокер.
 
### 3.5 Конфигурация и секреты

- Центральный класс `Settings` загружает переменные окружения из `.env` через `pydantic_settings`.
- Пример значений см. в [.env.example](../../.env.example).
- Live режим управляется конфигами Hydra: `config/live/bybit_paper.yaml` и `config/live/bybit_live.yaml`.

---

## 4. План реализации

1. **Settings и .env** – создать `packages/core/settings.py` на `pydantic_settings`, добавить `.env.example`.
2. **Meta‑апдейт тренера** – сохранить `feature_stats` и параметры комиссий в артефакте, добавить поддержку `RecurrentPPO` в `inference.py`.
3. **Каркас live слоя** – создать `packages/live/` с моделями, брокером (базовый контракт, paper реализация), `stream.py` и `features_live.py`.
4. **Execution и Risk** – реализовать планирование ордеров и базовый риск‑движок с лимитами.
5. **Runner + CLI** – собрать цикл обработки данных и действия агента, добавить скрипты `live_paper.py` и `live_trade.py`.
6. **Bybit брокер** – реализовать REST/WS клиент с idempotency, реконнектом и reconcile.
7. **Наблюдаемость** – логирование, счётчики ошибок, PnL, обработка disconnect.
8. **Тесты** – покрыть модульные и интеграционные сценарии.

## 5. Какие файлы нужно изменить

- `packages/rl/trainer.py` — **update**: сохранение `feature_stats` и параметров `FeeModel`.
- `packages/rl/inference.py` — **update**: поддержка `RecurrentPPO` состояния.
- `packages/data/config.py` — **update**: ENV‑настройки для live режима.
- `packages/core/settings.py` — **add**: централизованная загрузка `.env` через `pydantic_settings`.
- `.env.example` — **add**: шаблон переменных окружения.
- `packages/data/config.py` — **update**: ENV‑настройки для live режима.
- `packages/backtest/fees.py` — **reuse**: подключение в `packages/live/broker/paper.py`.
- `packages/live/models.py` — **add**: датаклассы ордеров, позиций, котировок.
- `packages/live/broker/base.py` — **add**: контракт брокера.
- `packages/live/broker/bybit.py` — **add**: интеграция Bybit REST/WS.
- `packages/live/broker/paper.py` — **add**: симулятор брокера.
- `packages/live/stream.py` — **add**: поток `Kline` и `Quote`.
- `packages/live/features_live.py` — **add**: онлайн pipeline признаков.
- `packages/live/execution.py` — **add**: планирование ордеров.
- `packages/live/risk.py` — **add**: предторговые лимиты и kill‑switch.
- `packages/live/runner.py` — **add**: главный цикл.
- `scripts/live_paper.py` и `scripts/live_trade.py` — **add**: CLI для запуска.
- `config/live/bybit_paper.yaml` и `config/live/bybit_live.yaml` — **add**: hydra‑конфиги.
- `docs/features/live-trading-layer.md` — **add**: этот документ.
- `docs/live/*` и `docs/settings.md` — **add**: справочные документы по контрактам и классам.

---

## 6. Потенциальные риски

| Риск | Вероятность | Влияние | План |
|------|-------------|----------|------|
| Несоответствие порядка признаков | Средняя | Высокое | Сверять `meta.feature_stats.columns` с `features_live` при старте, падать при несовпадении |
| Потеря связи с Bybit WS | Высокая | Среднее | Реализовать reconnect и circuit breaker до синхронизации |
| Неверная нормировка в онлайне | Средняя | Среднее | Использовать сохранённые `mu/sd`; покрыть тестами совпадения |
| Ошибки исполнения ордеров из‑за шагов/минималок | Средняя | Высокое | `ExecutionPlanner` округляет к `qty_step` и проверяет `min_qty` |
| Превышение лимитов риска | Низкая | Высокое | `RiskEngine` блокирует сделки и активирует kill‑switch |

---

## 7. План тестирования

### 7.1 Unit-тесты
- `execution` – корректное формирование ордеров, обработка переворота позиции, `reduce_only`, `post_only` и округление количества.
- `risk` – проверка лимитов позиции, дневного убытка, price band и частоты ордеров.
- `features_live` – совпадение колонок и нормировки с оффлайн `feature_stats`.
- `broker.paper` – сценарии полного и частичного исполнения, комиссии.
  Юнит‑тесты располагаются в `tests/unit/...`, повторяя структуру соответствующих модулей.

### 7.2 Интеграционные тесты
- `runner` на записанном kline‑трека проверяет детерминированный набор ордеров и PnL.
- Рестарт процесса и reconcile позиций/ордеров.
- Обработка drop WS → fallback REST → пауза торговли до ресинка.
  Интеграционные тесты находятся в `tests/integration/...` и запускаются с флагом `--run-integration`.

### 7.3 Приёмочные сценарии

```gherkin
Scenario: Paper runner places and reconciles orders on Bybit feed
Given сохранённый набор свечей BTCUSDT
When запускается `scripts/live_paper.py` с моделью и конфигом
Then ордера отправляются в симулятор, PnL рассчитывается с учётом комиссий
```
