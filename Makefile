# Makefile for setting up the project environment

# Variables
DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/raw
INTERIM_DATA_DIR = $(DATA_DIR)/interim
PROCESSED_DATA_DIR = $(DATA_DIR)/processed
ENV_NAME = term_deposit_env
DATA_URL = 
DATA_FILE = $(RAW_DATA_DIR)/data.csv

# Targets
.PHONY: all setup_data download_data create_env

all: setup_data download_data create_env

setup_data:
	@echo "Creating data directories..."
	mkdir -p $(RAW_DATA_DIR) $(PROCESSED_DATA_DIR) $(INTERIM_DATA_DIR)

download_data: setup_data
	@echo "Downloading data..."
	curl -o $(DATA_FILE) $(DATA_URL)

create_env:
	@echo "Creating and activating environment..."
	conda create -n $(ENV_NAME) python=3.8 -y
	@echo "Environment $(ENV_NAME) created. Activate it using 'conda activate $(ENV_NAME)'"