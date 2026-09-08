PY := .venv/bin/python

.PHONY: help setup download preprocess eda train-baseline train-effnet train eval compare ablate app smoke clean all

help:
	@echo "make setup           create venv and install dependencies"
	@echo "make download        fetch the APTOS 2019 dataset"
	@echo "make preprocess      preprocess images and build splits"
	@echo "make smoke           2-epoch end-to-end sanity run"
	@echo "make train           train baseline then efficientnet"
	@echo "make eda             dataset figures for the report"
	@echo "make eval            evaluate every trained model on the test split"
	@echo "make compare         build the model comparison table + figure"
	@echo "make ablate          sweep optimizers (and activations) on the baseline"
	@echo "make app             launch the Streamlit demo"

setup:
	python3.14 -m venv .venv
	.venv/bin/pip install -r requirements.txt

download:
	$(PY) -m src.download_data

preprocess:
	$(PY) -m src.preprocess

smoke:
	$(PY) -m src.train --model efficientnet --epochs 2 --limit 64

train-baseline:
	$(PY) -m src.train --model baseline

train-effnet:
	$(PY) -m src.train --model efficientnet

train: train-baseline train-effnet

eda:
	$(PY) -m src.eda

eval:
	$(PY) -m src.evaluate --model baseline  --split test
	$(PY) -m src.evaluate --model efficientnet --split test

compare:
	$(PY) -m src.compare --split test

ablate:
	$(PY) -m src.ablation --sweep optimizer --epochs 15
	$(PY) -m src.ablation --sweep activation --epochs 15

all: preprocess eda train eval compare

app:
	.venv/bin/streamlit run app/app.py

clean:
	rm -rf outputs/models/*.pt outputs/logs/* outputs/figures/*
