.PHONY: install test lint format pilot kill_experiments smoke clean

install:
	pip install -e .[test]

test:
	pytest tests/ -q

test-fast:
	pytest tests/ -q -x --ff

lint:
	ruff check sprl tests scripts

format:
	ruff format sprl tests scripts
	ruff check --fix sprl tests scripts

pilot:
	python scripts/run_pilot.py --config configs/pilot_200m_bf16.yaml --steps 4

kill_experiments:
	python scripts/run_kill_experiments.py --bet all --steps 4

smoke:
	python scripts/run_pilot.py --steps 2

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
