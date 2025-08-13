# Documentation


## Live trading layer

- [Data contracts](live/data-contracts.md)
- [Broker interface](live/broker.md)
- [Stream and features](live/stream-features.md)
- [Execution and risk](live/execution-risk.md)
- [Runner](live/runner.md)
- [Application settings](settings.md)

## Layers

- [Data and Backtesting](data.md)
- [TrainConfig](config.md)
- [Trainer](trainer.md)
- [Agents](agents.md)

## Development

Run checks before committing code:

```
pre-commit run --files <files>
make lint FILES="<files>"
make test
```
