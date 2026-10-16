# Makefile for PyChOpMarg project.
#
# Original author: David Banas <capn.freako@gmail.com>  
# Original date:   November 4, 2024
#
# Copyright (c) 2024 David Banas; all rights reserved World wide.

.PHONY: dflt help check format lint lint-fix type-check docs build upload upload_test test clean distclean

SRC_DIR := src/pychopmarg
DOCS_DIR := docs
UV_EXEC := uv
UVX_EXEC := uvx
PYVERS := 3.10 3.11 3.12 3.13
# One test environment per Python version (see `test`), so they don't replace each other.
VENV_DIR := $(CURDIR)/.venvs
PROJ_VER := $(shell ${UV_EXEC} version | cut -f 2 -d ' ') 
TEST_EXP ?= tests

# Put it first so that "make" without arguments is like "make help".
dflt: help

# Prevent implicit rule searching for makefiles.
$(MAKEFILE_LIST): ;

check:
	${UVX_EXEC} -w packaging>=24.2 validate-pyproject pyproject.toml

format:
	${UVX_EXEC} isort src/ tests/
	${UVX_EXEC} black src/ tests/

lint:
	${UVX_EXEC} ruff check ${SRC_DIR}
	${UVX_EXEC} flake8 --ignore=E501,E272,E241,E222,E221,F401,F403 ${SRC_DIR}

lint-fix:
	${UVX_EXEC} ruff check --fix ${SRC_DIR}
	${UVX_EXEC} flake8 --ignore=E501,E272,E241,E222,E221 ${SRC_DIR}

type-check:
	${UV_EXEC} run mypy --follow-untyped-imports ${SRC_DIR}

docs:
	pushd ${DOCS_DIR}; PROJ_VER=${PROJ_VER} ${UV_EXEC} run sphinx-build -j auto -b html . build/; popd

requirements.txt:
	${UV_EXEC} export --package pychopmarg --no-dev --no-hashes --no-emit-project -o requirements.txt

build: requirements.txt
	${UV_EXEC} build --clear --no-create-gitignore --out-dir dist

upload: build
	${UVX_EXEC} uv-publish --repo pypi dist/*

upload_test: build
	${UVX_EXEC} uv-publish --repo testpypi dist/*

test:
	@for VERSION in $(PYVERS); do \
        UV_PROJECT_ENVIRONMENT=$(VENV_DIR)/py$$VERSION $(UV_EXEC) run --python $$VERSION pytest -vv \
            --cov=pychopmarg --cov-report=html \
            --cov-report=term-missing $(TEST_EXP) || exit 1; \
    done

clean:
	rm -rf .tox build/ docs/build/ .mypy_cache .pytest_cache .venv $(VENV_DIR) src/*.egg-info

distclean: clean
	rm -rf dist/

help:
	@echo "Available targets:"
	@echo "  check: Validate the 'pyproject.toml' file."
	@echo "  format: Reformats the source code."
	@echo "    This will run EXTREME reformatting on the code. Use with caution!"
	@echo "  lint: Runs 'ruff' & 'flake8' on the source code."
	@echo "  lint-fix: Runs 'ruff' & 'flake8' on the source code, passing the '--fix' option to 'ruff'."
	@echo "  type-check: Checks the Python source code for type errors, by running 'mypy' over it."
	@echo "  docs: Generates the package documentation, by running 'sphinx' on the source code."
	@echo "    To view the resultant API documentation, open 'docs/build/index.html' in a browser."
	@echo "  requirements.txt: Generates a 'requirements.txt' file consistent with the current contents of 'uv.lock'."
	@echo "  build: Build the project wheel and source tarball files, for distribution."
	@echo "  upload: Uploads source tarball and wheel to PyPi."
	@echo "    (Only David Banas can do this.)"
	@echo "  upload_test: Uploads source tarball and wheel to TestPyPi."
	@echo "    (Only David Banas can do this.)"
	@echo "  test: Run testing for all supported Python versions, each in its own environment under '.venvs/'."
	@echo "    Define environment variable 'TEST_EXP' for focused testing."
	@echo "  clean: Remove all previous build results, virtual environments, and cache contents."
	@echo "  distclean: Run a normal 'clean' and remove all distribution targets."
