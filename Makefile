.PHONY: install dev test lint example

install:
	pip install -e ".[vision]"

dev:
	pip install -e ".[dev,tello,data]"

test:
	pytest -q

lint:
	ruff check .

example:
	python examples/04_shared_motion_decoder.py
