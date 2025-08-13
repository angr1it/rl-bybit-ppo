FILES ?=

.PHONY: lint test

lint:
	pre-commit run --files $(FILES)

test:
	pytest --cov=packages --cov-report=term-missing || true
