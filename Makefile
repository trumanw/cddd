.PHONY: rebuild pretrain
rebuild:
	pip uninstall cddd -y
	pip install .
pretrain:
	rm -rf example/default_model/runs/*
	python cddd/train.py --hparams_from_file True --hparams_file_name example/default_model/hparams.json