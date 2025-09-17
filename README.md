# TrafficInternVL — Quick Start

🏆 4th Place in AI City Challenge 2025 Track 2

TrafficInternVL provides a complete pipeline for traffic scene understanding: data prep (caption + VQA), training, export, and inference.

## Requirements
- Conda environment with Python 3.10 (recommended)
- CUDA 12.2 installed (driver/toolkit) for training
- Install dependencies with `pip install -r requirements.txt`
- Ensure dataset is downloaded locally (see below)
- Hardware: peak GPU memory ~86.5 GB (reduce batch/grad-accum if less)

## Project Structure

```
TrafficInternVL/
├── data-preparation/
│   ├── task1/
│   │   ├── data/
│   │   │   ├── BDD_PC_5k/
│   │   │   │   ├── annotations/
│   │   │   │   │   ├── bbox_annotated/
│   │   │   │   │   ├── bbox_generated/
│   │   │   │   │   └── caption/
│   │   │   │   └── videos/
│   │   │   ├── WTS/
│   │   │   │   ├── annotations/
│   │   │   │   │   ├── bbox_annotated/
│   │   │   │   │   ├── bbox_generated/
│   │   │   │   │   └── caption/
│   │   │   │   └── videos/
│   │   │   ├── SubTask1-Caption/
│   │   │   │   ├── WTS_DATASET_PUBLIC_TEST/
│   │   │   │   └── WTS_DATASET_PUBLIC_TEST_BBOX/
│   │   │   ├── SubTask2-VQA/
│   │   │   │   └── WTS_VQA_PUBLIC_TEST.json
│   │   │   └── wts_dataset_zip/
│   │   └── ... # python and shell scripts
│   └── task2/
│       ├── processed_data_subtask2_best_view/
│       │   ├── images/
│       │   │   ├── train/
│       │   │   └── test/
│       │   ├── wts_dataset_train_subtask2_best_view.jsonl
│       │   └── wts_dataset_test_subtask2_best_view.jsonl
│       └── ... # python and shell scripts
├── models-training/
│   └── data/
└── ... # other project files
```

## 1) Setup
```bash
cd <YOUR_PROJECT_PATH>/TrafficInternVL
conda create -n trafficinternvl python=3.10 -y
conda activate trafficinternvl
pip install -r requirements.txt
# Ensure your PyTorch build matches CUDA 12.2
```

## 2) Data Download & Paths
- Download WTS dataset: [WTS: Woven Traffic Safety Dataset](https://woven-visionai.github.io/wts-dataset-homepage/)
- Place datasets under:
  - `data-preparation/task1/data/`
- Update paths in scripts to your local project path:
```bash
# Example (task1 scripts)
root="<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/"
save_folder="<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/processed_anno/"
```

## 3) Task 1 — Caption Data
- Prepare training data:
```bash
cd data-preparation/task1
# Edit paths in prepare_data_train.sh first
./prepare_data_train.sh   # outputs: wts_bdd_local_train.json
```
- Convert to InternVL format and copy to training dir:
```bash
cd processed_anno/internvl_format/
# Edit paths in final_dataset.sh and update_prompt.sh if needed
./final_dataset.sh
./update_prompt.sh        # outputs: final_local_train_dataset.json
cp final_local_train_dataset.json ../../models-training/data/
```
- Prepare test data (optional):
```bash
cd data-preparation/task1
# Edit test_root & generate_test_frames_path in prepare_data_test.sh
./prepare_data_test.sh
```

## 4) Task 2 — VQA (Best View) Data
- Prerequisite: Task 1 data present in `data-preparation/task1/data/`
- Training split:
```bash
cd data-preparation/task2
bash run_prepare_data_train_subtask2_best_view.sh
# outputs:
# - processed_data_subtask2_best_view/wts_dataset_train_subtask2_best_view.jsonl
# - processed_data_subtask2_best_view/images/train/
```
- Test split:
```bash
cd data-preparation/task2
bash run_prepare_data_test_subtask2_best_view.sh
# outputs:
# - processed_data_subtask2_best_view/wts_dataset_test_subtask2_best_view.jsonl
# - processed_data_subtask2_best_view/images/test/
```

- Quick verification:
```bash
wc -l processed_data_subtask2_best_view/*.jsonl
find processed_data_subtask2_best_view/images -name "*.jpg" | wc -l
head -n 1 processed_data_subtask2_best_view/wts_dataset_train_subtask2_best_view.jsonl | python3 -m json.tool
```

## 5) Train
```bash
cd models-training
# Edit paths in setup_environment.sh if needed
./setup_environment.sh

# Edit paths/options in train.sh if needed
./train.sh
```
Notes:
- Default dataset name is set in `models-training/data/dataset_info.json` (should be `aicity_local_dataset`).
- Default model: `OpenGVLab/InternVL3-8B-hf` with LoRA rank 64.
- Default output dir: `model/InternVL-38B/r8-task1-github-test`.

## 6) Export & Inference
- Export merged model (base + LoRA):
```bash
cd models-training
# In export.sh set:
# --adapter_name_or_path=model/InternVL-38B/r8-task1-github-test
# --export_dir=./exported_models/your_exported_model
./export.sh
```
- Inference:
```bash
# In inference_image_input.sh set:
# --model_path=./exported_models/your_exported_model
# --test_data_dir=<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/generate_test_frames/bbox_local
./inference_image_input.sh
```

## Configuration Checklist
- Task 1 scripts:
  - `root`, `save_folder`
  - `test_root`, `generate_test_frames_path` in `prepare_data_test.sh`
- Task 2 scripts:
  - Verify `data-preparation/task1/data/` exists
  - Check `DATA_ROOT`, `TEST_VIDEOS_DIR`, `TEST_BBOX_DIR`, `VQA_TEST_FILE`, `OUTPUT_ROOT`
- Training scripts:
  - `output_dir` in `train.sh`
  - Dataset in `models-training/data/dataset_info.json`
- Export & inference scripts:
  - `--adapter_name_or_path`, `--export_dir`, `--model_path`, `--test_data_dir`

## Key Scripts
- Task 1 (Caption):
  - `data-preparation/task1/prepare_data_train.sh`
  - `data-preparation/task1/prepare_data_test.sh`
  - `data-preparation/task1/processed_anno/internvl_format/final_dataset.sh`
  - `data-preparation/task1/processed_anno/internvl_format/update_prompt.sh`
- Task 2 (VQA Best View):
  - `data-preparation/task2/run_prepare_data_train_subtask2_best_view.sh`
  - `data-preparation/task2/run_prepare_data_test_subtask2_best_view.sh`
  - `data-preparation/task2/prepare_data_train_subtask2_best_view.py`
  - `data-preparation/task2/prepare_data_test_subtask2_best_view.py`
- Training & Inference:
  - `models-training/setup_environment.sh`
  - `models-training/train.sh`
  - `models-training/export.sh`
  - `models-training/inference_image_input.sh`

## Troubleshooting (Quick)
- Paths: most errors are path-related—double-check the variables listed above.
- Environment: ensure your venv is active and `pip install -r requirements.txt` succeeded.
- Task 2 performance: reduce `NUM_WORKERS` if running out of memory.
- Terminal: use `bash` (not `sh`) for colored output and echo formatting.

## Reference
- CityLLaVA: Efficient Fine-Tuning for VLMs in City Scenario — [Paper (arXiv:2405.03194)](https://doi.org/10.48550/arXiv.2405.03194) · [Code (AliOpenTrek CityLLaVA)](https://github.com/alibaba/AICITY2024_Track2_AliOpenTrek_CityLLaVA)
- LLaMA-Factory — Unified Efficient Fine-Tuning of 100+ LLMs & VLMs: [GitHub](https://github.com/hiyouga/LLaMA-Factory)