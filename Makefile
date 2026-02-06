.PHONY: setup dev lint fmt test

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt -r requirements-dev.txt

dev:
	uvicorn main:app --reload --port 3000 --host 0.0.0.0

lint:
	ruff check .

fmt:
	black .

test:
	pytest
