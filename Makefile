.PHONY: install lint format typecheck test test-compliance test-unit docker-up docker-down docker-logs mlflow-ui clean help

help:
	@echo "Available targets:"
	@echo "  install          Install project with dev and test dependencies"
	@echo "  lint             Run ruff linter and formatter check"
	@echo "  format           Run ruff formatter and fix linter issues"
	@echo "  typecheck        Run mypy for static type checking"
	@echo "  test             Run all tests with coverage"
	@echo "  test-compliance  Run module compliance test suite"
	@echo "  test-unit        Run unit tests (excluding compliance suite)"
	@echo "  docker-up        Start services with docker compose"
	@echo "  docker-down      Stop docker compose services"
	@echo "  docker-logs      Follow docker compose logs"
	@echo "  mlflow-ui        Print mlflow UI URL"
	@echo "  clean            Remove cache and build artifacts"

install:
	pip install -e '.[dev,test]'

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy .

test:
	pytest tests/ -v --cov

test-compliance:
	pytest tests/compliance/ -v

test-unit:
	pytest tests/ -v --ignore=tests/compliance

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

mlflow-ui:
	@echo "Open http://localhost:5001 in your browser"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
