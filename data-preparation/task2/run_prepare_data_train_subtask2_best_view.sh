#!/bin/bash
# run_prepare_data_train_subtask2_best_view.sh - VQA TRAINING BEST VIEW + ENVIRONMENT PREPARATION SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
# • Runs the VQA training data preparation with best view selection + environment questions
# • Phase questions: Best view selection per phase (0-4) with bbox priority
# • Environment questions: Vehicle view priority + largest images, avoids frame 0
# • Max 2 images per sample: [crop, full] when bbox available or [full] when no bbox
# • No image resizing - preserves original resolution for better VLM training
# • Creates merged training dataset from train+val splits
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${WTS_DATA_ROOT:-datasets/wts_dataset_zip}"
OUT_ROOT="${WTS_OUT_ROOT:-processed_data_subtask2_best_view}"
WORKERS="${NUM_WORKERS:-64}"

echo "🚀 WTS VQA BEST VIEW + ENVIRONMENT TRAINING DATA PREPARATION"
echo "=" | head -c 70; echo
echo "📋 Configuration:"
echo "  • Script: ${SCRIPT_DIR}/prepare_data_train_subtask2_best_view.py"
echo "  • Data root: ${DATA_ROOT}"
echo "  • Output root: ${OUT_ROOT}"
echo "  • Workers: ${WORKERS}"
echo "📊 Processing Types:"
echo "  • Phase-based questions (0-4): Best view per phase with bbox priority"
echo "  • Environment questions: Vehicle view priority + largest images"
echo "  • Frame selection: Avoids frame 0 for environment questions"
echo "=" | head -c 70; echo

# Check if data directory exists
if [ ! -d "${DATA_ROOT}" ]; then
    echo "❌ Data directory not found: ${DATA_ROOT}"
    echo "   Please set WTS_DATA_ROOT environment variable or create the directory"
    exit 1
fi

# Create output directory
mkdir -p "${OUT_ROOT}"

# Set output paths
IMAGES_OUT="${OUT_ROOT}/images"
JSONL_OUT="${OUT_ROOT}/wts_dataset_vqa_best_view_train.jsonl"

echo "📂 Output paths:"
echo "  • Images: ${IMAGES_OUT}"  
echo "  • JSONL: ${JSONL_OUT}"
echo

# Run the VQA best view + environment training preparation
echo "🎯 Starting VQA best view + environment training data preparation..."
python3 "${SCRIPT_DIR}/prepare_data_train_subtask2_best_view.py" \
    --data_root "${DATA_ROOT}" \
    --out_root "${IMAGES_OUT}" \
    --out_jsonl "${JSONL_OUT}" \
    --split both \
    --merge_splits \
    --num_workers "${WORKERS}"

# Check results
if [ -f "${JSONL_OUT}" ]; then
    SAMPLE_COUNT=$(wc -l < "${JSONL_OUT}")
    echo
    echo "✅ VQA Best View + Environment Training Data Preparation Complete!"
    echo "📊 Results:"
    echo "  • Total samples (phases + environment): ${SAMPLE_COUNT}"
    echo "  • Training JSONL: ${JSONL_OUT}"
    echo "  • Images directory: ${IMAGES_OUT}"
    echo
    echo "🔍 Sample structure check:"
    head -n 1 "${JSONL_OUT}" | python3 -m json.tool | head -n 20
    echo "  (showing first 20 lines of sample structure...)"
    echo
    echo "📈 Sample breakdown estimation:"
    echo "  • Each scenario: ~5 phase samples + ~16 environment samples"
    echo "  • Phase samples: Best view per phase with interaction focus"
    echo "  • Environment samples: Vehicle view priority with context focus"
else
    echo "❌ Output file not created: ${JSONL_OUT}"
    exit 1
fi

echo "🎉 VQA Best View + Environment Training Dataset Ready!"
echo "   Use this JSONL file for VLM training: ${JSONL_OUT}"
echo "🌍 Dataset includes both phase-based and environment questions for comprehensive training!" 