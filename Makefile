.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: gingado docs

gingado: $(SRC)
	nbdev_build_lib
	touch gingado

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	rm -rf dist
	pip install build && python -m build .

clean:
	rm -rf dist