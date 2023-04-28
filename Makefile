default: | help

SHELL=/bin/bash
POETRY_VERSION=1.4.2

setup: ## setup poetry and install packages
	# bootstrap of poetry
	if ! command -v poetry > /dev/null 2>&1; then \
		echo 'poetry is not installed'; \
		curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3; \
		poetry --version; \
	fi;
	poetry install --no-interaction
	if [[ ! -d ./input ]]; then \
		mkdir ./input; \
		pip install kaggle; \
		kaggle competitions download -c vesuvius-challenge-ink-detection -p ./input; \
		unzip ./input/vesuvius-challenge-ink-detection.zip -d ./input/vesuvius-challenge-ink-detection; \
	fi;

lint: ## lint code
	poetry run pflake8 scripts src
	poetry run isort -c --diff scripts src
	poetry run black --check scripts src

mypy: ## typing check
	poetry run mypy --config-file pyproject.toml scirpts src

format: ## auto format
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive scripts src
	poetry run isort scripts src
	poetry run black scripts src

test: ## run test with pytest
	poetry run pytest -c tests

clean:
	rm -rf ./output/* wandb/*

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
