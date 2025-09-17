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
# WORKSPACE_ROOT is full path untill ~/TrafficInternVL
WORKSPACE_ROOT="$(pwd)/../.."
DATA_ROOT="${WTS_DATA_ROOT:-${WORKSPACE_ROOT}/data-preparation/task1/data/wts_dataset_zip}"
OUT_ROOT="${WTS_OUT_ROOT:-${WORKSPACE_ROOT}/data-preparation/task2/processed_data_subtask2_best_view}"
WORKERS="${NUM_WORKERS:-64}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}STARTING: WTS VQA BEST VIEW + ENVIRONMENT TRAINING DATA PREPARATION${NC}"
echo "=================================================================="
echo -e "${YELLOW}INFO: Configuration:${NC}"
echo "  • Script: prepare_data_train_subtask2_best_view.py"
echo "  • Data root: ${DATA_ROOT}"
echo "  • Output root: ${OUT_ROOT}"
echo "  • Workers: ${WORKERS}"
echo ""
echo -e "${PURPLE}TARGET: TRAINING DATA FEATURES:${NC}"
echo "  • PHASE QUESTIONS: Best view per phase (0-4) with bbox priority"
echo "  • ENVIRONMENT QUESTIONS: Vehicle view priority + largest images"  
echo "  • FRAME SELECTION: Avoids frame 0 for environment questions"
echo "  • IMAGE STRATEGY: [crop + full] with bbox OR [full] without bbox"
echo "  • RESOLUTION: No resize - preserves original resolution for VLM training"
echo "  • DATASET: Merges train+val splits for comprehensive training data"
echo ""
echo -e "${CYAN}STATS: Processing Logic:${NC}"
echo "  • PHASE QUESTIONS: Select best view per phase with interaction focus"
echo "  • ENVIRONMENT QUESTIONS: Vehicle view priority with context focus"  
echo "  • BBOX PRIORITY: Views with annotations get higher selection priority"
echo "  • IMAGE COUNT: Max 2 images per sample for optimal VLM training"
echo "  • SPLIT MERGING: Combines train and validation data for larger dataset"
echo "=================================================================="

# Validate input paths
echo -e "${BLUE}SEARCHING: Validating input paths...${NC}"

if [ ! -d "${DATA_ROOT}" ]; then
    echo -e "${RED}ERROR: Data directory not found: ${DATA_ROOT}${NC}"
    echo -e "${RED}   Please set WTS_DATA_ROOT environment variable or create the directory${NC}"
    exit 1
fi

echo -e "${GREEN}SUCCESS: Data directory validated${NC}"

# Check dataset structure
echo -e "${BLUE}STATS: Checking training dataset structure...${NC}"

# Count training videos
TRAIN_COUNT=$(find "${DATA_ROOT}" -name "*.mp4" -path "*/train/*" 2>/dev/null | wc -l || echo 0)
echo "  • Training videos: ${TRAIN_COUNT}"

# Count validation videos  
VAL_COUNT=$(find "${DATA_ROOT}" -name "*.mp4" -path "*/val/*" 2>/dev/null | wc -l || echo 0)
echo "  • Validation videos: ${VAL_COUNT}"

# Count bbox files
BBOX_COUNT=$(find "${DATA_ROOT}" -name "*_bbox.json" 2>/dev/null | wc -l || echo 0)
echo "  • Bbox annotation files: ${BBOX_COUNT}"

# Check VQA files
VQA_TRAIN_COUNT=$(find "${DATA_ROOT}" -name "*train*.json" -path "*VQA*" 2>/dev/null | wc -l || echo 0)
VQA_VAL_COUNT=$(find "${DATA_ROOT}" -name "*val*.json" -path "*VQA*" 2>/dev/null | wc -l || echo 0)
echo "  • VQA training files: ${VQA_TRAIN_COUNT}"
echo "  • VQA validation files: ${VQA_VAL_COUNT}"

echo ""

# Create output directory
echo -e "${BLUE}FOLDER: Setting up output directory...${NC}"
mkdir -p "${OUT_ROOT}/images"

# Set output paths
IMAGES_OUT="${OUT_ROOT}/images"
JSONL_OUT="${OUT_ROOT}/wts_dataset_train_subtask2_best_view.jsonl"

echo -e "${GREEN}SUCCESS: Output directory created: ${OUT_ROOT}${NC}"
echo ""
echo -e "${YELLOW}FOLDER: Output paths:${NC}"
echo "  • Images: ${IMAGES_OUT}"  
echo "  • JSONL: ${JSONL_OUT}"
echo ""

# Run the VQA best view + environment training preparation
echo -e "${BLUE}PROCESSING: Starting VQA best view + environment training data preparation...${NC}"
echo "Start time: $(date)"

python3 prepare_data_train_subtask2_best_view.py \
    --data_root "${DATA_ROOT}" \
    --out_root "${IMAGES_OUT}" \
    --out_jsonl "${JSONL_OUT}" \
    --split both \
    --merge_splits \
    --num_workers "${WORKERS}"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"

# Check results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}COMPLETE: Processing completed successfully!${NC}"
    
    # Verify output
    if [ -f "${JSONL_OUT}" ]; then
        SAMPLE_COUNT=$(wc -l < "${JSONL_OUT}")
        echo -e "${GREEN}SUCCESS: Output JSONL created with ${SAMPLE_COUNT} samples${NC}"
        
        # Count images
        IMAGE_COUNT=$(find "${IMAGES_OUT}" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}SUCCESS: Generated ${IMAGE_COUNT} processed images${NC}"
        
        # Count different image types
        FULL_COUNT=$(find "${IMAGES_OUT}" -name "*_full.jpg" 2>/dev/null | wc -l || echo 0)
        CROP_COUNT=$(find "${IMAGES_OUT}" -name "*_crop.jpg" 2>/dev/null | wc -l || echo 0)
        VIEW_COUNT=$(find "${IMAGES_OUT}" -name "*_view*.jpg" 2>/dev/null | wc -l || echo 0)
        ENV_COUNT=$(find "${IMAGES_OUT}" -name "env_*.jpg" 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}SUCCESS: Generated ${FULL_COUNT} full images, ${CROP_COUNT} cropped images, ${VIEW_COUNT} multi-view images, ${ENV_COUNT} environment images${NC}"
        
        # Analyze training data results
        echo -e "${BLUE}STATS: Analyzing training data results:${NC}"
        python3 -c "
import json
from collections import Counter

with open('${JSONL_OUT}', 'r') as f:
    samples = [json.loads(line) for line in f]

# Analyze question types
question_types = []
split_types = []
num_images_counts = []

for sample in samples:
    question_type = sample.get('question_type', sample.get('phase_name', 'unknown'))
    question_types.append(question_type)
    
    # Check split (train/val)
    split_type = sample.get('split', 'unknown')
    split_types.append(split_type)
    
    # Count images per sample
    if 'conversations' in sample and sample['conversations']:
        conv = sample['conversations'][0]
        if isinstance(conv, list) and conv:
            content = conv[0].get('content', [])
            image_count = len([c for c in content if c['type'] == 'image'])
            num_images_counts.append(image_count)

# Question type distribution
if question_types:
    question_dist = Counter(question_types)
    print(f'  • Question type distribution: {dict(question_dist)}')

# Split distribution
if split_types:
    split_dist = Counter(split_types)
    print(f'  • Split distribution: {dict(split_dist)}')

# Images per sample
if num_images_counts:
    avg_images = sum(num_images_counts) / len(num_images_counts)
    max_images = max(num_images_counts)
    min_images = min(num_images_counts)
    print(f'  • Average images per question: {avg_images:.1f}')
    print(f'  • Max images per question: {max_images}')
    print(f'  • Min images per question: {min_images}')
    
    # Image count distribution
    dist = Counter(num_images_counts)
    print(f'  • Image count distribution: {dict(sorted(dist.items()))}')
"
        
        # Show sample structure
        echo ""
        echo -e "${BLUE}SEARCHING: Sample structure check:${NC}"
        head -n 1 "${JSONL_OUT}" | python3 -m json.tool | head -n 20
        echo "  (showing first 20 lines of sample structure...)"
        
        # Show sample directories and files
        echo ""
        echo -e "${YELLOW}FOLDER: Sample output directories:${NC}"
        find "${IMAGES_OUT}" -maxdepth 2 -type d | head -5
        
        # Show sample images
        echo -e "${YELLOW}SAVED: Sample images:${NC}"
        echo "  Environment images:"
        find "${IMAGES_OUT}" -name "env_*.jpg" | head -3
        echo "  Phase images (full):"
        find "${IMAGES_OUT}" -name "*_full.jpg" | head -3
        echo "  Phase images (crop):"
        find "${IMAGES_OUT}" -name "*_crop.jpg" | head -3
        
        echo ""
        echo -e "${BLUE}INFO: Training Data Summary:${NC}"
        echo "  • Total samples: ${SAMPLE_COUNT}"
        echo "  • Total images: ${IMAGE_COUNT}"
        echo "  • Full images: ${FULL_COUNT}"
        echo "  • Cropped images: ${CROP_COUNT}"
        echo "  • Multi-view images: ${VIEW_COUNT}"
        echo "  • Environment images: ${ENV_COUNT}"
        echo "  • Strategy: Best view selection with bbox priority + merged train/val splits"
        echo "  • Output JSONL: ${JSONL_OUT}"
        echo "  • Images directory: ${IMAGES_OUT}"
        
    else
        echo -e "${RED}ERROR: Output JSONL file not created${NC}"
        exit 1
    fi
else
    echo -e "${RED}ERROR: Processing failed with exit code ${EXIT_CODE}${NC}"
    exit $EXIT_CODE
fi

echo -e "${GREEN}FINISHED: VQA Best View + Environment Training Dataset Ready!${NC}"
echo -e "${CYAN}TARGET: Use this JSONL file for VLM training: ${JSONL_OUT}${NC}"
echo -e "${PURPLE}ENVIRONMENT: Dataset includes both phase-based and environment questions for comprehensive training!${NC}" 