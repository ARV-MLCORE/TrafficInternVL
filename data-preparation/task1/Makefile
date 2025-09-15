SHELL := /bin/bash

ROOT_DIR := /home/deepzoom/arv-aicity2-data/Park/AICITY2024_Track2_AliOpenTrek_CityLLaVA/data_preprocess/data
SAVE_DIR := ./data/processed_anno
FINAL_DATA_DIR := ./data
SRC_DIR := ./src/data_preprocessing
VENV_PYTHON := ./venv/bin/python

.PHONY: all clean

all: prepare_data

prepare_data:
	@echo "Starting data preparation..."
	@mkdir -p $(SAVE_DIR)/frame_bbox_anno
	@mkdir -p $(SAVE_DIR)/llava_format
	$(VENV_PYTHON) $(SRC_DIR)/extract_wts_frame_bbox_anno.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/frame_bbox_anno --split train
	$(VENV_PYTHON) $(SRC_DIR)/extract_wts_frame_bbox_anno.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/frame_bbox_anno --split val
	$(VENV_PYTHON) $(SRC_DIR)/extract_bdd_frame_bbox_anno.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/frame_bbox_anno --split train
	$(VENV_PYTHON) $(SRC_DIR)/extract_bdd_frame_bbox_anno.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/frame_bbox_anno --split val
	$(VENV_PYTHON) $(SRC_DIR)/transform_llava_format.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/llava_format --split train --output-tag local --wts-global-image-path $(ROOT_DIR)/WTS/bbox_global --bdd-global-image-path $(ROOT_DIR)/BDD_PC_5k/bbox_global --wts-local-image-path $(ROOT_DIR)/WTS/bbox_local --bdd-local-image-path $(ROOT_DIR)/BDD_PC_5k/bbox_local
	$(VENV_PYTHON) $(SRC_DIR)/transform_llava_format.py --root $(ROOT_DIR) --save-folder $(SAVE_DIR)/llava_format --split val --wts-global-image-path $(ROOT_DIR)/WTS/bbox_global --bdd-global-image-path $(ROOT_DIR)/BDD_PC_5k/bbox_global --output-tag global
	$(VENV_PYTHON) $(SRC_DIR)/add_stage_prompt.py --input-file $(SAVE_DIR)/llava_format/wts_bdd_local_train.json --bbox-anno-file $(SAVE_DIR)/frame_bbox_anno/wts_train_all_video_with_bbox_anno_first_frame.json --output-file $(SAVE_DIR)/llava_format/wts_bdd_local_train_staged.json
	$(VENV_PYTHON) $(SRC_DIR)/update_prompt.py --input-file $(SAVE_DIR)/llava_format/wts_bdd_local_train_staged.json --output-file $(SAVE_DIR)/llava_format/wts_bdd_local_train_updated.json
	$(VENV_PYTHON) $(SRC_DIR)/finalize_dataset.py --input-file $(SAVE_DIR)/llava_format/wts_bdd_local_train_updated.json --output-file $(FINAL_DATA_DIR)/final_local_train_dataset.json
	@echo "Data preparation finished."

clean:
	@echo "Cleaning up..."
	@rm -rf $(SAVE_DIR)
	@rm -f $(FINAL_DATA_DIR)/final_local_train_dataset.json
	@echo "Cleanup finished."