default: | help

SHELL=/bin/bash
POETRY_VERSION=1.4.0

setup: ## setup poetry and install packages
	# bootstrap of poetry
	if ! command -v poetry > /dev/null 2>&1; then \
		echo 'poetry is not installed'; \
		curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3; \
		poetry --version; \
	fi;
	poetry install --no-interaction

lint: ## lint code
	poetry run pflake8 scirpts src
	poetry run isort -c --diff scirpts src
	poetry run black --check scirpts src

mypy: ## typing check
	poetry run mypy --config-file pyproject.toml scirpts src

format: ## auto format
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive scirpts src
	poetry run isort scirpts src
	poetry run black scirpts src

test: ## run test with pytest
	poetry run pytest -c tests

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
