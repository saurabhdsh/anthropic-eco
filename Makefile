.PHONY: offline session models test setup

setup:
	bash scripts/setup.sh

offline:
	python session.py --offline --auto

session:
	python session.py

models:
	python session.py --list-models

test:
	python -m unittest discover -s tests -v
