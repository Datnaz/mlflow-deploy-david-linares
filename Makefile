# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
train:
	python train.py
validate:
	Python validate.py

all: train validate