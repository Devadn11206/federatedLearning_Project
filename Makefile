.PHONY: install verify-env clean download data data-real baselines baselines-real train ablations app report demo

PYTHON := .venv/Scripts/python
PIP := .venv/Scripts/pip
PYTEST := .venv/Scripts/pytest

install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

verify-env:
	$(PYTEST) tests/test_phase0.py -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

download:
	$(PYTHON) data/download_datasets.py --dataset all --resize 224 --cleanup-raw

data:
	$(PYTHON) data/prepare_consortium.py --synthetic-dryrun

data-real:
	$(PYTHON) data/prepare_consortium.py --mode real

baselines:
	$(PYTHON) experiments/train_baselines.py --synthetic-dryrun --epochs 1

baselines-real:
	$(PYTHON) experiments/train_baselines.py --mode real --epochs 5

train:
	$(PYTHON) fl_engine/run_simulation.py --rounds 15 --local-epochs 1

ablations:
	$(PYTHON) experiments/run_ablations.py

app:
	$(PYTHON) -m streamlit run app/dashboard.py

report:
	$(PYTHON) reports/generate_final_report.py

demo:
	$(PYTHON) -m streamlit run app/dashboard.py
