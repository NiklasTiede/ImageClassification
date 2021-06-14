
.PHONY: training evaluation

# ------------- Performing Experiments  ------------------------------------

training: ##
	python train.py

evaluation:
	python eval_conf_matrix.py



# ------------- Test/Lint  ------------------------------------

.PHONY: pre-commit pre-commit-update lint test test-all

pre-commit: ## apply pre-commit to all files
	pre-commit run --all-files

pre-commit-update: ## update pre-commit
	pre-commit autoupdate

test: ## run tests quickly with the default Python
	pytest -s -vvv

test-all: ## test across Python 3.6 - 3.9
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source covid19pyclient -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html



# ------------- Clean Artifacts (Test/Lint/Build) ---------------------

.PHONY: clean clean-pyc clean-test clean-build

clean: clean-pyc clean-test clean-build ## remove all build, test and Python artifacts

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -fr .mypy_cache
	rm -fr .pytest_cache



# ------------  Help  --------------------------------------

.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
