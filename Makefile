.PHONY: rebuild
rebuild:
	pip uninstall cddd -y
	python setup.py install