.PHONY: install uninstall

install:
	pip install -e .

uninstall:
	pip uninstall -y cax-scripts
