.PHONY: clean clean-test clean-pyc clean-build docs help test dev-container dev-build dev-start
.DEFAULT_GOAL := help

# ... (previous Makefile content) ...
#
#
#

# Development container commands
dev-build:
	docker-compose build stata-dev

dev-start:
	docker-compose run --rm stata-dev

dev-stop:
	docker-compose down

dev-shell:
	docker-compose exec stata-dev /bin/bash

dev-test:
	docker-compose run --rm stata-dev pytest

dev-lint:
	docker-compose run --rm stata-dev black src/stata tests
	docker-compose run --rm stata-dev isort src/stata tests
	docker-compose run --rm stata-dev flake8 src/stata tests

dev-install:
	docker-compose run --rm stata-dev pip install -e ".[dev]"

# Add Jupyter notebook support
jupyter:
	docker-compose run --rm -p 8888:8888 stata-dev jupyter lab --ip 0.0.0.0 --allow-root --no-browser

# Development workflow shortcuts
dev-setup: dev-build dev-install

dev-update:
	git pull
	docker-compose build stata-dev
	docker-compose run --rm stata-dev pip install -e ".[dev]"