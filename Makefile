# Makefile for PyChOpMarg project.
#
# Original author: David Banas <capn.freako@gmail.com>  
# Original date:   November 4, 2024
#
# Copyright (c) 2024 David Banas; all rights reserved World wide.

.PHONY: dflt help check format lint lint-fix type-check docs build upload upload_test test clean distclean

# PROJ_NAME := PyChOpMarg
# PROJ_FILE := pyproject.toml
# PROJ_INFO := src/PyChOpMarg.egg-info/PKG-INFO
# VER_FILE := .proj_ver
# VER_GETTER := ./get_proj_ver.py
# SHELL_EXEC := bash
# PYTHON_EXEC := python -I
# TOX_EXEC := tox
# TOX_SKIP_ENV := format
# PYVERS := 310 311 312 313
# PLATFORMS := lin mac win
# NOTEBOOK_DIR := notebook
# NOTEBOOKS := PyChOpMarg_vs_MATLAB
# NOTEBOOK_EXT := .ipynb
# REPORT_EXT := .html
# PYTHON_SRCS := $(wildcard src/pychopmarg/*.py src/pychopmarg/*/*.py)

SRC_DIR := src/pychopmarg
DOCS_DIR := docs
UV_EXEC := uv
UVX_EXEC := uvx
PYVERS := 3.10 3.11 3.12 3.13
PROJ_VER := $(shell ${UV_EXEC} version | cut -f 2 -d ' ') 
TEST_EXP ?= tests

# Put it first so that "make" without arguments is like "make help".
dflt: help

# Prevent implicit rule searching for makefiles.
$(MAKEFILE_LIST): ;

# ${VER_FILE}: ${PROJ_INFO}
# 	${PYTHON_EXEC} ${VER_GETTER} ${PROJ_NAME} $@

# ${PROJ_INFO}: ${PROJ_FILE}
# 	${PYTHON_EXEC} -m build
# 	${PYTHON_EXEC} -m pip install -e .

reports: ${NOTEBOOK_DIR}/$(addsuffix ${REPORT_EXT},${NOTEBOOKS})

%${REPORT_EXT}: %${NOTEBOOK_EXT}
	${TOX_EXEC} run -e report -- $<

%${NOTEBOOK_EXT}: ${PYTHON_SRCS}
	${TOX_EXEC} run -e notebook -- $@

# tox:
# 	TOX_SKIP_ENV="${TOX_SKIP_ENV}" ${TOX_EXEC} -m test

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
	pushd ${DOCS_DIR}; PROJ_VER=${PROJ_VER} ${UV_EXEC} run sphinx-build -j auto -b html source/ build/; popd

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
        $(UV_EXEC) run --python $$VERSION pytest -vv \
            --cov=pyibisami --cov-report=html \
            --cov-report=term-missing $(TEST_EXP); \
    done

clean:
	rm -rf .tox build/ docs/build/ .mypy_cache .pytest_cache .venv src/*.egg-info

distclean: clean
	rm -rf dist/

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
	@echo "  reports: Generate reports from selected Jupyter notebooks."
	@echo "    To view the resultant reports, open 'notebook/<report_name>.html' in a browser."
