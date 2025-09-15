#!/bin/bash
# run_prepare_data_test_subtask2_best_view.sh - COMPREHENSIVE PROCESSING VERSION  
# Script to run SubTask2 VQA test data preparation with intelligent image selection
# KEY FEATURES:
# 1. ENVIRONMENT QUESTIONS: Uses 1 largest image from all available views
# 2. EVENT PHASE QUESTIONS: Selects the BEST available view for each phase
# 3. Prioritizes views with bounding box annotations for event phases
# 4. Provides appropriate image counts: Environment (1) | Event phases (2)
# 5. Intelligent view ranking based on annotation quality and view type

set -e  # Exit on any error

# Configuration
WORKSPACE_ROOT="/workspace"
TEST_VIDEOS_DIR="${WORKSPACE_ROOT}/datasets/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST"
TEST_BBOX_DIR="${WORKSPACE_ROOT}/datasets/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST_BBOX"
VQA_TEST_FILE="${WORKSPACE_ROOT}/datasets/SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json"
OUTPUT_ROOT="${WORKSPACE_ROOT}/processed_data_subtask2_best_view"
OUTPUT_JSONL="${OUTPUT_ROOT}/wts_dataset_test_subtask2_best_view.jsonl"
NUM_WORKERS=32

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 WTS Dataset SubTask2 VQA Test Processor - COMPREHENSIVE PROCESSING${NC}"
echo "=================================================================="
echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  • Test videos: ${TEST_VIDEOS_DIR}"
echo "  • Test bbox: ${TEST_BBOX_DIR}"  
echo "  • VQA test file: ${VQA_TEST_FILE}"
echo "  • Output root: ${OUTPUT_ROOT}"
echo "  • Output JSONL: ${OUTPUT_JSONL}"
echo "  • Workers: ${NUM_WORKERS}"
echo ""
echo -e "${PURPLE}🎯 COMPREHENSIVE PROCESSING FEATURES:${NC}"
echo "  • ENVIRONMENT QUESTIONS: Uses largest image + largest pedestrian bbox from all views"
echo "  • EVENT PHASE QUESTIONS: Smart view ranking with bbox priority"
echo "  • BBOX PRIORITY: Views with annotations get +10 priority points"
echo "  • VIEW TYPE PRIORITY: overhead_view(3) > vehicle_view(2) > single_view(1)"
echo "  • ADAPTIVE OUTPUT: WITH BBOX: 1 view → crop + full | WITHOUT BBOX: 2 views → both full"
echo "  • NO RESIZE: Full images keep original resolution (no 896x896 resize)"
echo "  • INTELLIGENT CROPPING: Only when bbox available, adaptive scaling"
echo "  • METADATA TRACKING: Records selection reasoning and image types"
echo ""
echo -e "${CYAN}📊 Processing Logic:${NC}"
echo "  • ENVIRONMENT: Find largest image + largest pedestrian bbox from all views"
echo "  • EVENT PHASES: Discover all available camera views for each phase"
echo "  • PRIORITY CALC: bbox_priority(0/10) + view_type_priority(0-3)"
echo "  • VIEW SELECTION: Select view with highest combined priority score"
echo "  • IMAGE EXTRACTION: Environment: largest + pedestrian bbox | Phase: crop + full OR 2 full"
echo "  • METADATA: Include selection reasoning and image types in samples"
echo ""
echo -e "${YELLOW}📸 Output Format per Question:${NC}"
echo "  • ENVIRONMENT: env_largest_{camera}_view1.jpg + env_pedestrian_{camera}_view2.jpg (2 images)"
echo "  • EVENT PHASE WITH BBOX: best_view_{camera}_phase{N}_f{frame}_full.jpg + _crop.jpg"
echo "  • EVENT PHASE WITHOUT BBOX: best_view_{camera}_phase{N}_f{frame}_view1.jpg + _view2.jpg"
echo "  • Environment: 2 images | Event phases: 2 images per sample"
echo "=================================================================="

# Validate input paths  
echo -e "${BLUE}🔍 Validating input paths...${NC}"

if [ ! -d "${TEST_VIDEOS_DIR}" ]; then
    echo -e "${RED}❌ Test videos directory not found: ${TEST_VIDEOS_DIR}${NC}"
    exit 1
fi

if [ ! -d "${TEST_BBOX_DIR}" ]; then
    echo -e "${RED}❌ Test bbox directory not found: ${TEST_BBOX_DIR}${NC}"
    exit 1
fi

if [ ! -f "${VQA_TEST_FILE}" ]; then
    echo -e "${RED}❌ VQA test file not found: ${VQA_TEST_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All input paths validated${NC}"

# Check dataset structure
echo -e "${BLUE}📊 Checking dataset structure...${NC}"

# Count normal_trimmed videos  
NORMAL_COUNT=$(find "${TEST_VIDEOS_DIR}/videos/test/public/normal_trimmed" -name "*.mp4" 2>/dev/null | wc -l || echo 0)
echo "  • Normal trimmed videos: ${NORMAL_COUNT}"

# Count WTS public videos
WTS_COUNT=$(find "${TEST_VIDEOS_DIR}/videos/test/public/" -maxdepth 3 -name "*.mp4" 2>/dev/null | wc -l || echo 0)
echo "  • WTS public videos: ${WTS_COUNT}"

# Count external videos
EXTERNAL_COUNT=$(find "${TEST_VIDEOS_DIR}/external/BDD_PC_5K/videos/test/public" -name "*.mp4" 2>/dev/null | wc -l || echo 0)
echo "  • External BDD videos: ${EXTERNAL_COUNT}"

# Count bbox files
BBOX_COUNT=$(find "${TEST_BBOX_DIR}" -name "*_bbox.json" 2>/dev/null | wc -l || echo 0)
echo "  • Bbox annotation files: ${BBOX_COUNT}"

# Check VQA file
VQA_ENTRIES=$(python -c "import json; data=json.load(open('${VQA_TEST_FILE}')); print(len(data))" 2>/dev/null || echo "unknown")
echo "  • VQA test entries: ${VQA_ENTRIES}"

# Show sample scenarios with comprehensive analysis
echo -e "${BLUE}📊 Sample scenario analysis (environment + event phases):${NC}"
python -c "
import json
from pathlib import Path

with open('${VQA_TEST_FILE}', 'r') as f:
    data = json.load(f)

test_videos_dir = Path('${TEST_VIDEOS_DIR}')
bbox_dir = Path('${TEST_BBOX_DIR}')

# Count different scenario types
env_only = sum(1 for s in data if s.get('conversations') and not s.get('event_phase'))
phase_only = sum(1 for s in data if s.get('event_phase') and not s.get('conversations'))
both = sum(1 for s in data if s.get('conversations') and s.get('event_phase'))

print(f'  📊 Scenario Distribution:')
print(f'    • Environment-only scenarios: {env_only}')
print(f'    • Event-phase-only scenarios: {phase_only}')  
print(f'    • Both types: {both}')
print(f'    • Total scenarios: {len(data)}')
print()

# Show sample environment scenario
env_sample = None
for i, scenario in enumerate(data):
    if scenario.get('conversations') and not scenario.get('event_phase'):
        env_sample = (i, scenario)
        break

if env_sample:
    i, scenario = env_sample
    video_file = scenario.get('videos', [''])[0]
    env_questions = len(scenario.get('conversations', []))
    print(f'  🌍 Sample Environment Scenario {i}:')
    print(f'    Video: {video_file}')
    print(f'    Environment questions: {env_questions}')
    print(f'    → Will use LARGEST image + LARGEST PEDESTRIAN BBOX from all available views')
    print()

# Show sample multi-phase scenario
phase_sample = None
for i, scenario in enumerate(data):
    if len(scenario.get('event_phase', [])) > 1:
        phase_sample = (i, scenario)
        break

if phase_sample:
    i, scenario = phase_sample
    video_file = scenario.get('videos', [''])[0]
    event_phases = scenario.get('event_phase', [])
    print(f'  📋 Sample Event Phase Scenario {i}:')
    print(f'    Video: {video_file}')
    print(f'    Event phases: {len(event_phases)}')
    
    # Check available cameras for this scenario
    if 'normal_' in video_file:
        scenario_dir = test_videos_dir / 'videos' / 'test' / 'public' / 'normal_trimmed' / Path(video_file).stem
    elif video_file.startswith('video') and len(video_file.split('_')) == 1:
        scenario_dir = test_videos_dir / 'external' / 'BDD_PC_5K' / 'videos' / 'test' / 'public'
        print(f'    Source: External BDD_PC_5K dataset')
    else:
        scenario_name = video_file.replace('.mp4', '').split('_192.168')[0] + '_T1'
        scenario_dir = test_videos_dir / 'videos' / 'test' / 'public' / scenario_name
    
    if scenario_dir.exists():
        if 'external' in str(scenario_dir):
            print(f'    Available cameras: [single_view] (external dataset)')
        else:
            cameras = [d.name for d in scenario_dir.iterdir() if d.is_dir()]
            print(f'    Available cameras: {cameras}')
    
    # Show best view selection for each phase
    for j, phase in enumerate(event_phases[:3]):  # Show first 3 phases
        phase_name = phase.get('labels', ['unknown'])[0]
        start_time = phase.get('start_time', 0)
        end_time = phase.get('end_time', 0)
        num_questions = len(phase.get('conversations', []))
        duration = float(end_time) - float(start_time)
        print(f'    Phase {j} ({phase_name}): {duration:.2f}s, {num_questions} questions')
        print(f'      → Best view selection: bbox availability + view type priority')
        
        # Check if bbox data exists for this scenario
        video_stem = Path(video_file).stem
        bbox_files = []
        if bbox_dir.exists():
            bbox_files = list(bbox_dir.rglob(f'*{video_stem}*_bbox.json'))
        
        if bbox_files:
            print(f'      → Found {len(bbox_files)} bbox files - HIGH PRIORITY views')
        else:
            print(f'      → No bbox files found - will use time-based selection')
"

echo ""

# Create output directory
echo -e "${BLUE}📁 Setting up output directory...${NC}"
mkdir -p "${OUTPUT_ROOT}/images"
echo -e "${GREEN}✅ Output directory created: ${OUTPUT_ROOT}${NC}"

# Run the processing script
echo -e "${BLUE}🎬 Starting SubTask2 VQA test data processing (comprehensive processing)...${NC}"
echo "Start time: $(date)"

python prepare_data_test_subtask2_best_view.py \
  --test_videos_dir "${TEST_VIDEOS_DIR}" \
  --test_bbox_dir "${TEST_BBOX_DIR}" \
  --vqa_test_file "${VQA_TEST_FILE}" \
  --out_root "${OUTPUT_ROOT}" \
  --out_jsonl "${OUTPUT_JSONL}" \
  --num_workers ${NUM_WORKERS}

EXIT_CODE=$?

echo ""
echo "End time: $(date)"

# Check results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 Processing completed successfully!${NC}"
    
    # Verify output
    if [ -f "${OUTPUT_JSONL}" ]; then
        SAMPLE_COUNT=$(wc -l < "${OUTPUT_JSONL}")
        echo -e "${GREEN}✅ Output JSONL created with ${SAMPLE_COUNT} samples${NC}"
        
        # Count images
        IMAGE_COUNT=$(find "${OUTPUT_ROOT}/images" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}✅ Generated ${IMAGE_COUNT} processed images${NC}"
        
        # Count different image types
        FULL_COUNT=$(find "${OUTPUT_ROOT}/images" -name "*_full.jpg" 2>/dev/null | wc -l || echo 0)
        CROP_COUNT=$(find "${OUTPUT_ROOT}/images" -name "*_crop.jpg" 2>/dev/null | wc -l || echo 0)
        VIEW_COUNT=$(find "${OUTPUT_ROOT}/images" -name "*_view*.jpg" 2>/dev/null | wc -l || echo 0)
        ENV_COUNT=$(find "${OUTPUT_ROOT}/images" \( -name "env_largest_*.jpg" -o -name "env_pedestrian_*.jpg" \) 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}✅ Generated ${FULL_COUNT} full images, ${CROP_COUNT} cropped images, ${VIEW_COUNT} multi-view images, ${ENV_COUNT} environment images${NC}"
        
        # Analyze comprehensive selection results
        echo -e "${BLUE}📊 Analyzing selection results (environment + event phases):${NC}"
        python -c "
import json
from collections import Counter

with open('${OUTPUT_JSONL}', 'r') as f:
    samples = [json.loads(line) for line in f]

# Analyze both event phase and environment selections
question_types = []
selection_types = []
bbox_counts = []
num_images_counts = []

for sample in samples:
    question_type = sample.get('question_type', sample.get('phase_name', 'unknown'))
    question_types.append(question_type)
    
    # Handle both best_view_info (event phases) and environment_info (environment questions)
    if 'best_view_info' in sample:
        info = sample['best_view_info']
        selection_types.append(info.get('selection_type', 'unknown'))
        bbox_counts.append(info.get('has_bbox', False))
        num_images_counts.append(info.get('num_images', 0))
    elif 'environment_info' in sample:
        info = sample['environment_info']
        selection_types.append(info.get('selection_type', 'unknown'))
        bbox_counts.append(False)  # Environment questions don't use bbox
        num_images_counts.append(info.get('num_images', 0))
    else:
        selection_types.append('unknown')
        bbox_counts.append(False)
        num_images_counts.append(0)

# Question type distribution
if question_types:
    question_dist = Counter(question_types)
    print(f'  • Question type distribution: {dict(question_dist)}')

# Selection type distribution
if selection_types:
    selection_dist = Counter(selection_types)
    bbox_ratio = sum(bbox_counts) / len(bbox_counts) * 100 if bbox_counts else 0
    avg_images = sum(num_images_counts) / len(num_images_counts) if num_images_counts else 0
    
    print(f'  • Selection type distribution: {dict(selection_dist)}')
    print(f'  • Samples with bbox data: {bbox_ratio:.1f}%')
    print(f'  • Average images per sample: {avg_images:.1f}')
    
    # Images count distribution
    images_dist = Counter(num_images_counts)
    print(f'  • Images per sample distribution: {dict(sorted(images_dist.items()))}')

# Images per sample analysis from actual conversation content
image_counts = []
for sample in samples:
    conv = sample['conversations'][0][0]
    image_count = len([c for c in conv['content'] if c['type'] == 'image'])
    image_counts.append(image_count)

if image_counts:
    avg_images = sum(image_counts) / len(image_counts)
    max_images = max(image_counts)
    min_images = min(image_counts)
    print(f'  • Average images per question: {avg_images:.1f}')
    print(f'  • Max images per question: {max_images}')
    print(f'  • Min images per question: {min_images}')
    
    # Image count distribution
    dist = Counter(image_counts)
    print(f'  • Image count distribution: {dict(sorted(dist.items()))}')
    
    # Check consistency - should be 1 or 2 images depending on question type
    env_samples = [s for s in samples if s.get('question_type') == 'environment']
    phase_samples = [s for s in samples if s.get('question_type') == 'vqa']
    
    if env_samples:
        env_image_counts = [len([c for c in s['conversations'][0][0]['content'] if c['type'] == 'image']) for s in env_samples]
        print(f'  • Environment questions: {len(env_samples)} samples, avg {sum(env_image_counts)/len(env_image_counts):.1f} images (expected: 2.0)')
    
    if phase_samples:
        phase_image_counts = [len([c for c in s['conversations'][0][0]['content'] if c['type'] == 'image']) for s in phase_samples]
        print(f'  • Event phase questions: {len(phase_samples)} samples, avg {sum(phase_image_counts)/len(phase_image_counts):.1f} images')
"
        
        # Show sample directories
        echo -e "${YELLOW}📂 Sample output directories:${NC}"
        find "${OUTPUT_ROOT}/images" -maxdepth 2 -type d | head -5
        
        # Show sample images
        echo -e "${YELLOW}📸 Sample images:${NC}"
        echo "  Environment images (largest):"
        find "${OUTPUT_ROOT}/images" -name "env_largest_*.jpg" | head -2
        echo "  Environment images (pedestrian):"
        find "${OUTPUT_ROOT}/images" -name "env_pedestrian_*.jpg" | head -2
        echo "  Event phase images (full):"
        find "${OUTPUT_ROOT}/images" -name "best_view_*_full.jpg" | head -2
        echo "  Event phase images (crop):"
        find "${OUTPUT_ROOT}/images" -name "best_view_*_crop.jpg" | head -2
        
        echo ""
        echo -e "${BLUE}📋 Comprehensive Processing Summary:${NC}"
        echo "  • Total samples: ${SAMPLE_COUNT}"
        echo "  • Total images: ${IMAGE_COUNT}"
        echo "  • Environment images (largest + pedestrian): ${ENV_COUNT}"
        echo "  • Event phase full images: ${FULL_COUNT}"
        echo "  • Event phase cropped images: ${CROP_COUNT}"
        echo "  • Event phase multi-view images: ${VIEW_COUNT}"
        echo "  • Environment strategy: largest image + largest pedestrian bbox per scenario"
        echo "  • Event phase strategy: Best view selection (bbox priority)"
        echo "  • Output JSONL: ${OUTPUT_JSONL}"
        echo "  • Images directory: ${OUTPUT_ROOT}/images"
        
        # Calculate expected images based on question types
        python -c "
import json
with open('${OUTPUT_JSONL}', 'r') as f:
    samples = [json.loads(line) for line in f]

env_questions = sum(1 for s in samples if s.get('question_type') == 'environment')
phase_questions = sum(1 for s in samples if s.get('question_type') == 'vqa')

# Environment: 2 images per question, Event phase: 2 images per question  
expected_images = env_questions * 2 + phase_questions * 2
actual_images = ${IMAGE_COUNT}

print(f'Expected images: {expected_images} (env: {env_questions}×2 + phase: {phase_questions}×2)')
print(f'Actual images: {actual_images}')
print(f'Match: {expected_images == actual_images}')
"
        
    else
        echo -e "${RED}❌ Output JSONL file not created${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Processing failed with exit code ${EXIT_CODE}${NC}"
    exit $EXIT_CODE
fi

echo -e "${GREEN}🏁 SubTask2 VQA test data preparation (comprehensive processing) completed successfully!${NC}"
echo -e "${CYAN}🎯 Comprehensive processing ensures optimal image selection for both environment and event phase questions!${NC}" 