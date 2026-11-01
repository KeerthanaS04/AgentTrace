# Thin wrapper over tasks.py so `make <target>` and `python tasks.py <target>` behave the same.
PY ?= python

.PHONY: setup test lint format serve eval smoke retrieval

setup:
	$(PY) tasks.py setup

test:
	$(PY) tasks.py test

lint:
	$(PY) tasks.py lint

format:
	$(PY) tasks.py format

serve:
	$(PY) tasks.py serve

eval:
	$(PY) tasks.py eval

smoke:
	$(PY) tasks.py smoke

retrieval:
	$(PY) tasks.py retrieval
