# Makefile for PyChOpMarg project.
#
# Original author: David Banas <capn.freako@gmail.com>  
# Original date:   November 4, 2024
#
# Copyright (c) 2024 David Banas; all rights reserved World wide.

.PHONY: dflt help check tox format lint flake8 type-check docs build upload test clean etags conda-build conda-skeleton chaco enable pyibis-ami pyibis-ami-dev pybert pybert-dev etags

PROJ_NAME := PyChOpMarg
PROJ_FILE := pyproject.toml
PROJ_INFO := src/PyChOpMarg.egg-info/PKG-INFO
VER_FILE := .proj_ver
VER_GETTER := ./get_proj_ver.py
PYTHON_EXEC := python -I
TOX_EXEC := tox
TOX_SKIP_ENV := format
PYVERS := 310 311 312 313
PLATFORMS := lin mac win

# Put it first so that "make" without arguments is like "make help".
dflt: help

check:
	${TOX_EXEC} run -e check

${VER_FILE}: ${PROJ_INFO}
	${PYTHON_EXEC} ${VER_GETTER} ${PROJ_NAME} $@

${PROJ_INFO}: ${PROJ_FILE}
	${PYTHON_EXEC} -m build
	${PYTHON_EXEC} -m pip install -e .

tox:
	TOX_SKIP_ENV="${TOX_SKIP_ENV}" ${TOX_EXEC} -m test

format:
	${TOX_EXEC} run -e format

lint:
	${TOX_EXEC} run -e lint

flake8:
	${TOX_EXEC} run -e flake8

type-check:
	${TOX_EXEC} run -e type-check

docs: ${VER_FILE}
	source $< && ${TOX_EXEC} run -e docs

build: ${VER_FILE}
	${TOX_EXEC} run -e build

upload: ${VER_FILE}
	source $< && ${TOX_EXEC} run -e upload

test:
	@for V in ${PYVERS}; do \
		for P in ${PLATFORMS}; do \
			${TOX_EXEC} run -e "py$$V-$$P"; \
		done; \
	done

clean:
	rm -rf .tox docs/build/ .mypy_cache .pytest_cache .venv

help:
	@echo "Available targets:"
	@echo "  tox: Run all Tox environments."
	@echo "  check: Validate the 'pyproject.toml' file."
	@echo "  format: Run Tox 'format' environment."
	@echo "    This will run EXTREME reformatting on the code. Use with caution!"
	@echo "  lint: Run Tox 'lint' environment. (Runs 'pylint' on the source code.)"
	@echo "  flake8: Run Tox 'flake8' environment. (Runs 'flake8' on the source code.)"
	@echo "  type-check: Run Tox 'type-check' environment. (Runs 'mypy' on the source code.)"
	@echo "  docs: Run Tox 'docs' environment. (Runs 'sphinx' on the source code.)"
	@echo "    To view the resultant API documentation, open 'docs/build/index.html' in a browser."
	@echo "  build: Run Tox 'build' environment."
	@echo "    Builds source tarball and wheel, for installing or uploading to PyPi."
	@echo "  upload: Run Tox 'upload' environment."
	@echo "    Uploads source tarball and wheel to PyPi."
	@echo "    (Only David Banas can do this.)"
	@echo "  test: Run Tox testing for all supported Python versions."
	@echo "  clean: Remove all previous build results, virtual environments, and cache contents."
