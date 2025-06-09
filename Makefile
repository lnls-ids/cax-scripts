# PACKAGE:=$(shell basename $(shell pwd))
ifeq ($(CONDA_PREFIX),)
$(error Please activate your conda/mamba environment before installing)
endif

PACKAGE := caxscripts
PIP ?= pip

install:
	$(PIP) install -e .
	git clean -fdX

uninstall:
	$(PIP) uninstall -y $(PACKAGE)