# TrafficInternVL - Setup and Training Guide

This repository contains the complete pipeline for training and inference using the TrafficInternVL model for traffic scene understanding.

## 📁 Project Structure

After setup, your project structure should look like this:

```
TrafficInternVL/
├── data-preparation/
│   └── task1/
│       ├── data/
│       │   ├── BDD_PC_5k/
│       │   │   ├── annotations/
│       │   │   │   ├── bbox_annotated/
│       │   │   │   ├── bbox_generated/
│       │   │   │   └── caption/
│       │   │   └── videos/
│       │   ├── WTS/
│       │   │   ├── annotations/
│       │   │   │   ├── bbox_annotated/
│       │   │   │   ├── bbox_generated/
│       │   │   │   └── caption/
│       │   │   └── videos/
│       │   └── test_part/
│       │       ├── view_used_as_main_reference_for_multiview_scenario.csv
│       │       ├── WTS_DATASET_PUBLIC_TEST/
│       │       └── WTS_DATASET_PUBLIC_TEST_BBOX/
│       └── ... # python and shell scripts
├── models-training/
│   └── data/
└── ... # other project files
```

## 🚀 Quick Start Guide

### Step 1: Download and Setup Data

1. **Download the dataset** and place it under:
   ```
   <YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/
   ```

2. **Update configuration paths** in the data preparation scripts:
   - Edit the data preparation scripts to update these paths to match your setup:
   ```bash
   root="<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/"
   save_folder="<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/processed_anno/"
   ```

### Step 2: Environment Setup

1. **Create and activate virtual environment:**
   ```bash
   cd <YOUR_PROJECT_PATH>/TrafficInternVL
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Prepare Training Data

1. **Run data preparation script:**
   ```bash
   cd data-preparation/task1
   # Update paths in the script first, then run:
   ./prepare_data_train.sh
   ```
   This will generate: `wts_bdd_local_train.json`

2. **Process annotations to InternVL format:**
   ```bash
   cd processed_anno/internvl_format/
   ./final_dataset.sh
   ./update_prompt.sh
   ```
   This will generate: `final_local_train_dataset.json`

3. **Copy dataset to training directory:**
   ```bash
   cp final_local_train_dataset.json ../../models-training/data/
   ```

4. **Update dataset configuration:**
   The dataset info is already configured in:
   ```
   models-training/data/dataset_info.json
   ```

### Step 4: Prepare Test Data

1. **Update test data paths** in `prepare_data_test.sh`:
   ```bash
   test_root="<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/test_part"
   generate_test_frames_path="./data/generate_test_frames"
   ```

2. **Run test data preparation:**
   ```bash
   cd data-preparation/task1
   ./prepare_data_test.sh
   ```

### Step 5: Training Setup

1. **Setup training environment:**
   ```bash
   cd models-training
   ./setup_environment.sh
   ```

2. **Start training:**
   ```bash
   # Edit train.sh to modify output_dir if needed, then run:
   ./train.sh
   ```
   
   **Note:** The training script uses:
   - Dataset: `aicity_local_dataset`
   - Model: `OpenGVLab/InternVL3-8B-hf`
   - LoRA fine-tuning with rank 64
   - Default output directory: `model/InternVL-38B/r8-task1-github-test`

### Step 6: Model Export and Inference

1. **Export trained model with LoRA adapter:**
   ```bash
   cd models-training
   # Update the script with your trained model path
   # Edit export.sh and set:
   # --adapter_name_or_path=model/InternVL-38B/r8-task1-github-test (or your custom output_dir)
   # --export_dir=./exported_models/your_exported_model
   ./export.sh
   ```

2. **Run inference:**
   ```bash
   # Edit inference_image_input.sh and update:
   # --model_path=./exported_models/your_exported_model
   # --test_data_dir=<YOUR_PROJECT_PATH>/TrafficInternVL/data-preparation/task1/data/generate_test_frames/bbox_local
   ./inference_image_input.sh
   ```

## 📋 Configuration Checklist

Before running the pipeline, make sure to update these paths in the respective files:

### Data Preparation Scripts:
- [ ] `root` path in data preparation scripts
- [ ] `save_folder` path in data preparation scripts
- [ ] `test_root` in `prepare_data_test.sh`
- [ ] `generate_test_frames_path` in `prepare_data_test.sh`

### Training Scripts:
- [ ] `output_dir` in `train.sh` (default: `model/InternVL-38B/r8-task1-github-test`)
- [ ] Dataset name in `dataset_info.json` (should be `aicity_local_dataset`)

### Export and Inference Scripts:
- [ ] `--adapter_name_or_path` in `export.sh` (should point to `model/InternVL-38B/r8-task1-github-test` or your custom output_dir)
- [ ] `--export_dir` in `export.sh`
- [ ] `--model_path` in `inference_image_input.sh`
- [ ] `--test_data_dir` in `inference_image_input.sh`

## 🔧 Key Files and Scripts

| File/Script | Purpose |
|-------------|---------|
| `data-preparation/task1/prepare_data_train.sh` | **Prepare training data** |
| `data-preparation/task1/prepare_data_test.sh` | **Prepare test data** |
| `data-preparation/task1/processed_anno/internvl_format/final_dataset.sh` | Convert to InternVL format |
| `data-preparation/task1/processed_anno/internvl_format/update_prompt.sh` | Update prompts |
| `models-training/setup_environment.sh` | Setup training environment |
| `models-training/train.sh` | **Main training script** |
| `models-training/data/dataset_info.json` | Dataset configuration |
| `models-training/export.sh` | Export trained model |
| `models-training/inference_image_input.sh` | Run inference |

## 📝 Notes

- Make sure to activate your virtual environment before running any scripts
- Update all paths to match your local setup (replace `<YOUR_PROJECT_PATH>` with your actual path)
- The training process may take several hours depending on your hardware
- Ensure you have sufficient disk space for the dataset and model checkpoints

## 🆘 Troubleshooting

- If you encounter path-related errors, double-check that all paths in the configuration files are correctly updated
- Make sure your virtual environment is activated when running scripts
- Verify that all required dependencies are installed via `requirements.txt`
- Check that the dataset is properly downloaded and placed in the correct directory structure

## 📊 Expected Outputs

- **Training data**: `wts_bdd_local_train.json`
- **Final training dataset**: `final_local_train_dataset.json`
- **Trained model**: Saved in your specified `output_dir`
- **Exported model**: Saved in your specified `export_dir`
- **Inference results**: Generated after running inference script

---

For any issues or questions, please refer to the individual script files for more detailed configuration options.