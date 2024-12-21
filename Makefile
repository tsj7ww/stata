.PHONY: clean clean-test clean-pyc clean-build docs help test dev-container dev-build dev-start
.DEFAULT_GOAL := help

# ... (previous Makefile content) ...

# Development container commands
dev-build:
	docker-compose build predml-dev

dev-start:
	docker-compose run --rm predml-dev

dev-stop:
	docker-compose down

dev-shell:
	docker-compose exec predml-dev /bin/bash

dev-test:
	docker-compose run --rm predml-dev pytest

dev-lint:
	docker-compose run --rm predml-dev black src/predml tests
	docker-compose run --rm predml-dev isort src/predml tests
	docker-compose run --rm predml-dev flake8 src/predml tests

dev-install:
	docker-compose run --rm predml-dev pip install -e ".[dev]"

# Add Jupyter notebook support
jupyter:
	docker-compose run --rm -p 8888:8888 predml-dev jupyter lab --ip 0.0.0.0 --allow-root --no-browser

# Development workflow shortcuts
dev-setup: dev-build dev-install

dev-update:
	git pull
	docker-compose build predml-dev
	docker-compose run --rm predml-dev pip install -e ".[dev]"