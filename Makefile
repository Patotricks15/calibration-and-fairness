.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/03_calibration_fairness.py -o notebooks/03_calibration_fairness.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=credit-venv --ExecutePreprocessor.timeout=600 notebooks/03_calibration_fairness.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info
