#!/usr/bin/env python3
# prepare_data_test_subtask2_best_view.py - VQA TEST DATA PREPARATION WITH BEST VIEW SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# • Adapted from prepare_data_test_subtask2.py for best view selection approach
# • Selects the best available view (prioritizing views with bbox annotations)
# • Provides both cropped and full versions for each sample like subtask1 best_view
# • Uses intelligent view selection based on bbox availability and view quality
# • Maintains same multiprocessing performance as original
# ─────────────────────────────────────────────────────────────────────────────

import argparse, json, cv2
from pathlib import Path
from tqdm import tqdm
import time
from multiprocessing import Pool, Manager, cpu_count
import logging
from concurrent.futures import as_completed
import numpy as np
import random
import re

# Disable OpenCV threading to avoid conflicts with multiprocessing
cv2.setNumThreads(1)
rng = random.Random(42)          # set once at module scope for reproducibility

# Setup logging for better performance monitoring
logging.basicConfig(level=logging.WARNING)

# ---------- 1. Config -----------------------------------------------------------------
GREEN = (0, 255,   0)      # pedestrian box  (BGR)
BLUE  = (255, 0,   0)      # vehicle box

PHASE_DESCRIPTIONS = {
    "prerecognition": ("The vehicle and pedestrian have not yet noticed each other; traffic proceeds normally."),
    "recognition": ("The vehicle or pedestrian first becomes aware of the other or of a traffic hazard."),
    "judgement": ("The vehicle and pedestrian evaluate the risk and decide how to respond."),
    "action": ("The vehicle or pedestrian begins to determine appropriate actions to start the chosen manoeuvre."),
    "avoidance": ("The vehicle and pedestrian adjust motion until collision is averted or avoidance fails.")
}

# Map phase numbers to phase names
PHASE_ORDER = {
    0: "prerecognition",
    1: "recognition", 
    2: "judgement",
    3: "action",
    4: "avoidance"
}

# View priority for best view selection (higher number = higher priority)
VIEW_PRIORITY = {
    "overhead_view": 3,
    "vehicle_view": 2,
    "single_view": 1,
    "other_view": 0
}

# --------------------------------------------------------------------------------------

# ---------- 2. Helper functions (same as original + best view selection) --------------
def coco2map_fast(fp):
    """Fast COCO JSON → {phase_num: (frame_id, x1,y1,x2,y2)} with optimized I/O"""
    if not fp.exists():
        return {}
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
        
        # WTS dataset: each bbox file has exactly 5 annotations (one per phase)
        # Map by phase_number directly instead of frame_id
        result = {}
        for ann in data.get("annotations", []):
            phase_num = int(ann["phase_number"])  # Use explicit phase number
            frame_id = int(ann["image_id"])
            bbox = (ann["bbox"][0], ann["bbox"][1], 
                   ann["bbox"][0] + ann["bbox"][2], 
                   ann["bbox"][1] + ann["bbox"][3])
            result[phase_num] = (frame_id, bbox)
        
        return result
    except (json.JSONDecodeError, KeyError, IOError):
        return {}

def enlarge_fast(b, s, W, H):
    """Optimized bbox enlargement with minimal allocations"""
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * s, (y2 - y1) * s
    hw, hh = w * 0.5, h * 0.5
    return (max(0, int(cx - hw)), max(0, int(cy - hh)),
            min(W - 1, int(cx + hw)), min(H - 1, int(cy + hh)))

def enlarge_bbox_for_crop(bbox, scale=1.5):
    """Enlarge bbox for cropping - adapted from reference implementation"""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    center_x, center_y = x1 + width / 2, y1 + height / 2
    
    new_width = width * scale
    new_height = height * scale
    
    # Make square for better VLM guidance
    new_size = max(new_width, new_height)
    new_width = new_height = new_size
    
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    return new_x1, new_y1, new_x2, new_y2

def calculate_combined_bbox(bbox1, bbox2):
    """Calculate combined bbox from two bounding boxes"""
    x1_min = min(bbox1[0], bbox2[0])
    y1_min = min(bbox1[1], bbox2[1])
    x2_max = max(bbox1[2], bbox2[2])
    y2_max = max(bbox1[3], bbox2[3])
    
    return x1_min, y1_min, x2_max, y2_max

def constrain_bbox_within_frame(bbox, frame_shape):
    """Constrain bbox to stay within frame boundaries"""
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    return x1, y1, x2, y2

def resize_if_needed(image, max_size=896):
    """Resize image to max_size x max_size if either dimension exceeds max_size"""
    h, w = image.shape[:2]
    if w > max_size or h > max_size:
        resized_image = cv2.resize(image, (max_size, max_size), interpolation=cv2.INTER_AREA)
        print(f"      🔄 Resized from {w}x{h} to {max_size}x{max_size}")
        return resized_image
    return image

def extract_and_save_frame_best_view(video_path, frame_id, output_path, box_p=None, box_v=None, view_name=None, source="main", should_crop=False):
    """Enhanced frame extraction - saves either full or cropped based on bbox availability"""
    try:
        from decord import VideoReader
        vr = VideoReader(str(video_path))
        if frame_id >= len(vr):
            frame_id = len(vr) - 1
        frame_np = vr[frame_id].asnumpy()
        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        H, W = frame.shape[:2]
        
        # Draw bbox overlays on frame - ALWAYS draw bboxes when available
        has_bbox = False
        if box_p:
            # Draw original bbox without enlargement for better accuracy
            x1, y1, x2, y2 = box_p
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 3)
            has_bbox = True
            
        if box_v:
            # Draw original bbox without enlargement for better accuracy
            x1, y1, x2, y2 = box_v
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BLUE, 3)
            has_bbox = True

        # Ensure output directories exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # If should_crop is True and we have bboxes, create cropped version
        if should_crop and has_bbox:
            # Calculate combined bbox for cropping
            combined_bbox = None
            
            if box_p and box_v:
                # Both pedestrian and vehicle bboxes exist
                combined_bbox = calculate_combined_bbox(box_p, box_v)
            elif box_p:
                # Only pedestrian bbox
                combined_bbox = box_p
            elif box_v:
                # Only vehicle bbox  
                combined_bbox = box_v
            
            if combined_bbox is not None:
                # Calculate crop parameters with intelligent scaling
                width, height = combined_bbox[2] - combined_bbox[0], combined_bbox[3] - combined_bbox[1]
                area_ratio = (width * height) / (W * H)
                
                # Adaptive scaling based on area ratio
                min_area, max_area = 0.1, 0.6
                if area_ratio < min_area:
                    scale = 2.0  # Small objects need more context
                elif area_ratio > max_area:
                    scale = 1.2  # Large objects need less expansion
                else:
                    scale = 1.5  # Default scale
                
                # Handle very elongated bboxes
                if width / height > 4 or height / width > 4:
                    scale = min(scale, 1.3)  # Limit scaling for elongated objects
                
                # Enlarge and constrain crop bbox
                crop_bbox = enlarge_bbox_for_crop(combined_bbox, scale)
                crop_x1, crop_y1, crop_x2, crop_y2 = constrain_bbox_within_frame(crop_bbox, frame.shape)
                
                # Extract cropped region
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Redraw bboxes on cropped frame with adjusted coordinates
                if box_p:
                    # Adjust pedestrian bbox coordinates relative to crop
                    adj_x1 = max(0, int(box_p[0] - crop_x1))
                    adj_y1 = max(0, int(box_p[1] - crop_y1))
                    adj_x2 = min(cropped_frame.shape[1], int(box_p[2] - crop_x1))
                    adj_y2 = min(cropped_frame.shape[0], int(box_p[3] - crop_y1))
                    if adj_x2 > adj_x1 and adj_y2 > adj_y1:  # Valid bbox within crop
                        cv2.rectangle(cropped_frame, (adj_x1, adj_y1), (adj_x2, adj_y2), GREEN, 3)
                
                if box_v:
                    # Adjust vehicle bbox coordinates relative to crop
                    adj_x1 = max(0, int(box_v[0] - crop_x1))
                    adj_y1 = max(0, int(box_v[1] - crop_y1))
                    adj_x2 = min(cropped_frame.shape[1], int(box_v[2] - crop_x1))
                    adj_y2 = min(cropped_frame.shape[0], int(box_v[3] - crop_y1))
                    if adj_x2 > adj_x1 and adj_y2 > adj_y1:  # Valid bbox within crop
                        cv2.rectangle(cropped_frame, (adj_x1, adj_y1), (adj_x2, adj_y2), BLUE, 3)
                
                # Save cropped version if valid
                if cropped_frame.size > 0 and cropped_frame.shape[0] > 10 and cropped_frame.shape[1] > 10:
                    cv2.imwrite(str(output_path), cropped_frame)
                    print(f"      📸 Saved CROP {view_name}: ({cropped_frame.shape[1]}x{cropped_frame.shape[0]})")
                    return W, H, True  # Success with crop
                else:
                    print(f"      ⚠️  Invalid crop, saving full for {view_name}")
                    # Save full frame without resize as fallback
                    cv2.imwrite(str(output_path), frame)
                    return W, H, False  # Success but no crop
        
        # Save full frame without resize
        cv2.imwrite(str(output_path), frame)
        crop_type = "CROP" if should_crop else "FULL"
        print(f"      📸 Saved {crop_type} {view_name}: ({W}x{H})")
        return W, H, should_crop and has_bbox
        
    except Exception as e:
        print(f"❌ Error extracting frame from {video_path}: {e}")
        return None, None, False

def select_best_views_for_phase(phase_num, scenario_videos, video_base_dir, bbox_source_data, source, phase_data, original_scenario):
    """Select best views for a phase: 1 view with bbox (crop+full) OR 2 best views without bbox (both full)"""
    
    candidate_views = []
    
    # Collect all available views with their priorities
    if source == "external":
        # External data: single video file
        for video_file in scenario_videos:
            video_path = video_base_dir / video_file
            if not video_path.exists():
                continue
                
            camera_stem = Path(video_file).stem
            view_name = "single_view"
            
            # Get annotated frame for this phase (external data)
            frame_choice = get_annotated_frame_for_phase(phase_num, camera_stem, bbox_source_data, source)
            
            if frame_choice and (frame_choice.get('ped_box') or frame_choice.get('veh_box')):
                # Has valid bbox - add single view for crop+full approach
                bbox_priority = 10
                view_priority = VIEW_PRIORITY.get(view_name, 0)
                total_priority = bbox_priority + view_priority
                
                candidate_views.append({
                    'video_path': video_path,
                    'camera_stem': camera_stem,
                    'view_name': view_name,
                    'frame_choice': frame_choice,
                    'priority': total_priority,
                    'has_bbox': True
                })
            else:
                # No valid bbox - generate 2 time-based views for variety
                start_time = float(phase_data.get("start_time", 0))
                end_time = float(phase_data.get("end_time", 0))
                
                try:
                    from decord import VideoReader
                    vr = VideoReader(str(video_path))
                    fps = vr.get_avg_fps()
                    video_duration = len(vr) / fps
                    
                    # Generate 2 different time points for variety
                    duration = end_time - start_time
                    if duration > 1.0:
                        # Use start and middle of phase
                        time_points = [start_time, start_time + duration * 0.6]
                    else:
                        # Use start time and slightly later
                        time_points = [start_time, min(start_time + 0.5, video_duration - 0.1)]
                    
                    for i, time_point in enumerate(time_points):
                        frame_id = int(time_point * fps)
                        frame_id = min(frame_id, len(vr) - 1)
                        frame_id = max(1, frame_id)  # Avoid frame 0
                        
                        time_frame_choice = {
                            'frame': frame_id,
                            'ped_box': None,
                            'veh_box': None,
                        }
                        
                        # Add priority variation for ordering
                        view_priority = VIEW_PRIORITY.get(view_name, 0) + (1 - i * 0.1)  # Slight priority difference
                        
                        candidate_views.append({
                            'video_path': video_path,
                            'camera_stem': f"{camera_stem}_t{i+1}",  # Distinguish time points
                            'view_name': view_name,
                            'frame_choice': time_frame_choice,
                            'priority': view_priority,
                            'has_bbox': False
                        })
                        
                except Exception as e:
                    print(f"      ⚠️  Failed to process external video {video_file}: {e}")
                    continue
    
    else:
        # WTS scenarios with multiple camera views
        vehicle_view_processed = False
        
        for video_file in scenario_videos:
            scenario_name = extract_scenario_name(video_file)
            scenario_dir = video_base_dir / scenario_name
            
            if not scenario_dir.exists():
                continue
            
            # Process ALL available camera views
            for camera_dir in scenario_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                
                camera_view = camera_dir.name
                if "overhead" in camera_view.lower():
                    view_type = "overhead_view"
                elif "vehicle" in camera_view.lower():
                    view_type = "vehicle_view"
                    # Skip if already processed vehicle view
                    if vehicle_view_processed:
                        continue
                    vehicle_view_processed = True
                else:
                    view_type = "other_view"
                
                # Camera stem should match bbox file naming convention
                video_stem = Path(video_file).stem
                
                if "vehicle" in camera_view.lower():
                    camera_stem = f"{video_stem}_vehicle_view"
                    scenario_name = extract_scenario_name(video_file)
                    vehicle_video_name = f"{scenario_name}_vehicle_view.mp4"
                    video_path = camera_dir / vehicle_video_name
                else:
                    camera_stem = video_stem
                    video_path = camera_dir / video_file
                
                if not video_path.exists():
                    continue
                
                # Get annotated frame for this phase
                frame_choice = get_annotated_frame_for_phase(phase_num, camera_stem, bbox_source_data, source)
                
                if not frame_choice:
                    # Use time-based frame selection when no annotations available
                    start_time = float(phase_data.get("start_time", 0))
                    
                    try:
                        from decord import VideoReader
                        vr = VideoReader(str(video_path))
                        fps = vr.get_avg_fps()
                        video_duration = len(vr) / fps
                        
                        # Handle different video durations for vehicle_view vs overhead_view
                        if "vehicle" in camera_view.lower():
                            if start_time > video_duration:
                                proportion = phase_num / 4.0
                                adjusted_time = proportion * video_duration * 0.8
                                time_based_frame = int(adjusted_time * fps)
                            else:
                                time_based_frame = int(start_time * fps)
                        else:
                            time_based_frame = int(start_time * fps)
                        
                        time_based_frame = min(time_based_frame, len(vr) - 1)
                        time_based_frame = max(1, time_based_frame)  # Avoid frame 0
                        
                        frame_choice = {
                            'frame': time_based_frame,
                            'ped_box': None,
                            'veh_box': None,
                            'resize_896': True
                        }
                    except Exception as e:
                        continue
                
                # Calculate priority: bbox availability + view type priority
                bbox_priority = 10 if (frame_choice.get('ped_box') or frame_choice.get('veh_box')) else 0
                view_priority = VIEW_PRIORITY.get(view_type, 0)
                total_priority = bbox_priority + view_priority
                
                candidate_views.append({
                    'video_path': video_path,
                    'camera_stem': camera_stem,
                    'view_name': view_type,
                    'frame_choice': frame_choice,
                    'priority': total_priority,
                    'has_bbox': bbox_priority > 0
                })
    
    # Select views based on bbox availability
    if candidate_views:
        # Sort by priority (highest first)
        candidate_views.sort(key=lambda x: x['priority'], reverse=True)
        
        # Check if any view has bbox
        views_with_bbox = [v for v in candidate_views if v['has_bbox']]
        
        if views_with_bbox:
            # Use the best view with bbox - will generate crop + full
            best_view = views_with_bbox[0]
            print(f"    🎯 Selected best view WITH BBOX: {best_view['camera_stem']} ({best_view['view_name']}) - Priority: {best_view['priority']}")
            return [best_view], True  # Return single view, has_bbox=True
        else:
            # No bbox available - select top 2 views for variety
            selected_views = candidate_views[:2]  # Take top 2
            print(f"    🎯 Selected 2 best views WITHOUT BBOX:")
            for i, view in enumerate(selected_views):
                print(f"      {i+1}. {view['camera_stem']} ({view['view_name']}) - Priority: {view['priority']}")
            return selected_views, False  # Return multiple views, has_bbox=False
    
    return [], False

def get_annotated_frame_for_phase(phase_num, camera_stem, bbox_data, source="main"):
    """Get best annotated frame for a phase with intelligent fallback"""
    # Same implementation as original
    if source == "external":
        combined_anno_data = bbox_data.get('combined_anno', {})
        combined_gen_data = bbox_data.get('combined_gen', {})
        
        ped_box = veh_box = None
        frame_id = None
        
        # Try bbox_annotated first - only if frame and valid bbox exist
        if (camera_stem in combined_anno_data and 
            phase_num in combined_anno_data[camera_stem] and 
            combined_anno_data[camera_stem][phase_num] is not None):
            
            frame_id, bbox = combined_anno_data[camera_stem][phase_num]
            if frame_id is not None and bbox and len(bbox) >= 4:
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                area = width * height
                # Higher threshold for external data: 50x50 minimum and area > 3000
                if width >= 50 and height >= 50 and area >= 3000:
                    ped_box = bbox
                    print(f"      ✅ External bbox_annotated accepted for {camera_stem} phase {phase_num}: {width}x{height} (area: {area})")
                else:
                    print(f"      ⚠️  External bbox_annotated rejected for {camera_stem} phase {phase_num}: {width}x{height} (area: {area}) - too small")
                    frame_id = None  # Reset to try generated data
            else:
                frame_id = None  # No valid frame/bbox, try generated data
        
        # If bbox_annotated didn't work, try bbox_generated
        if (frame_id is None and 
            camera_stem in combined_gen_data and 
            phase_num in combined_gen_data[camera_stem] and 
            combined_gen_data[camera_stem][phase_num] is not None):
            
            frame_id, bbox = combined_gen_data[camera_stem][phase_num]
            if frame_id is not None and bbox and len(bbox) >= 4:
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                area = width * height
                # Higher threshold for external data: 50x50 minimum and area > 3000
                if width >= 50 and height >= 50 and area >= 3000:
                    ped_box = bbox
                    print(f"      ✅ External bbox_generated accepted for {camera_stem} phase {phase_num}: {width}x{height} (area: {area})")
                else:
                    print(f"      ⚠️  External bbox_generated rejected for {camera_stem} phase {phase_num}: {width}x{height} (area: {area}) - too small")
                    frame_id = None  # Reset if bbox too small
            else:
                frame_id = None  # No valid frame/bbox
            
        if frame_id is not None:
            return {
                'frame': frame_id,
                'ped_box': ped_box,
                'veh_box': veh_box
            }
            
    else:
        ped_anno_data = bbox_data.get('ped_anno', {})
        veh_anno_data = bbox_data.get('veh_anno', {})
        ped_gen_data = bbox_data.get('ped_gen', {})
        veh_gen_data = bbox_data.get('veh_gen', {})
        
        frame_choices = []
        
        if camera_stem in ped_anno_data and phase_num in ped_anno_data[camera_stem]:
            frame_id, ped_box = ped_anno_data[camera_stem][phase_num]
            veh_box = None
            
            if camera_stem in veh_anno_data and phase_num in veh_anno_data[camera_stem]:
                veh_frame_id, veh_box = veh_anno_data[camera_stem][phase_num]
                if veh_frame_id != frame_id:
                    veh_box = None
            
            frame_choices.append({
                'frame': frame_id,
                'ped_box': ped_box,
                'veh_box': veh_box,
                'source': 'ped_anno'
            })
        
        if camera_stem in veh_anno_data and phase_num in veh_anno_data[camera_stem]:
            frame_id, veh_box = veh_anno_data[camera_stem][phase_num]
            ped_box = None
            
            if camera_stem in ped_anno_data and phase_num in ped_anno_data[camera_stem]:
                ped_frame_id, ped_box = ped_anno_data[camera_stem][phase_num]
                if ped_frame_id != frame_id:
                    ped_box = None
            
            frame_choices.append({
                'frame': frame_id,
                'ped_box': ped_box,
                'veh_box': veh_box,
                'source': 'veh_anno'
            })
        
        if frame_choices:
            both_box_choices = [fc for fc in frame_choices if fc['ped_box'] and fc['veh_box']]
            if both_box_choices:
                return both_box_choices[0]
            else:
                return frame_choices[0]
        
        if camera_stem in ped_gen_data and phase_num in ped_gen_data[camera_stem]:
            frame_id, ped_box = ped_gen_data[camera_stem][phase_num]
            veh_box = None
            
            if camera_stem in veh_gen_data and phase_num in veh_gen_data[camera_stem]:
                veh_frame_id, veh_box = veh_gen_data[camera_stem][phase_num]
                if veh_frame_id != frame_id:
                    veh_box = None
            
            return {
                'frame': frame_id,
                'ped_box': ped_box,
                'veh_box': veh_box
            }
        
        if camera_stem in veh_gen_data and phase_num in veh_gen_data[camera_stem]:
            frame_id, veh_box = veh_gen_data[camera_stem][phase_num]
            ped_box = None
            
            if camera_stem in ped_gen_data and phase_num in ped_gen_data[camera_stem]:
                ped_frame_id, ped_box = ped_gen_data[camera_stem][phase_num]
                if ped_frame_id != frame_id:
                    ped_box = None
            
            return {
                'frame': frame_id,
                'ped_box': ped_box,
                'veh_box': veh_box
            }
    
    return None

def create_vqa_system_prompt(phase_name, phase_description):
    """Create an enhanced system prompt for VQA task with comprehensive guidance"""
    return f"""You are an expert traffic safety analyst specializing in vehicle-pedestrian interaction analysis. 
Your task is to analyze traffic scene images and answer multiple-choice questions about environment, pedestrian, and vehicle.

Current Scenario Context:
This image shows a vehicle-pedestrian interaction during the {phase_name} phase: {phase_description}

Visual Analysis Guidelines:
1. Bounding Boxes: Pay attention to the colored bounding boxes in the image:
   - GREEN bounding boxes highlight pedestrians
   - BLUE bounding boxes highlight vehicles
   - Note: Bounding boxes may not appear in all views of the same scene

2. Best View Analysis: This image represents the best available view for this scenario, selected based on annotation quality and view clarity.

3. Traffic Context: Consider traffic rules, road infrastructure, pedestrian crossings, traffic signals, and environmental factors that influence the interaction.

4. Safety Assessment: Evaluate potential risks, collision possibilities, and the appropriateness of actions taken by both pedestrians and vehicles.

Question Format:
- Each question is multiple-choice with options A, B, C, and D (Note that: some questions may have fewer than 4 options)
- Choose the single best answer based on your visual analysis

Analysis Approach:
1. First, identify all key elements in the scene (pedestrians, vehicles, infrastructure)
2. Understand the spatial relationships and movement patterns
3. Select the most accurate answer based on evidence from the image"""

def format_vqa_question(question_data, view_context="best_view"):
    """Format a VQA question with multiple choice options - test version without correct answer"""
    question = question_data["question"]
    choices = []
    
    # Collect available choices (a, b, c, d)
    for choice_key in ['a', 'b', 'c', 'd']:
        if choice_key in question_data and question_data[choice_key].strip():
            choice_letter = choice_key.upper()
            choice_text = question_data[choice_key]
            choices.append(f"{choice_letter}. {choice_text}")
    
    # Format the question with choices
    if choices:
        choices_text = "\n".join(choices)
        formatted_question = f"{question}\n{choices_text}"
    else:
        formatted_question = question
    
    return formatted_question

def process_test_data_fast(test_videos_dir, test_bbox_dir, vqa_test_file, out_root, out_jsonl, num_workers=32):
    """Main processing function for test data with best view selection"""
    test_videos_dir = Path(test_videos_dir)
    test_bbox_dir = Path(test_bbox_dir)
    vqa_test_file = Path(vqa_test_file)
    
    # Verify directories and files exist
    if not test_videos_dir.exists():
        print(f"❌ Test videos directory not found: {test_videos_dir}")
        return 0
    
    if not test_bbox_dir.exists():
        print(f"❌ Test bbox directory not found: {test_bbox_dir}")
        return 0
        
    if not vqa_test_file.exists():
        print(f"❌ VQA test file not found: {vqa_test_file}")
        return 0

    # Load VQA test data
    print(f"📂 Loading VQA test data from {vqa_test_file}")
    with open(vqa_test_file, 'r') as f:
        vqa_test_data = json.load(f)
    
    print(f"🔍 Found {len(vqa_test_data)} test scenarios")

    # Pre-load bbox data for fast lookups (same as original)
    print(f"📦 Loading bbox data from {test_bbox_dir}")
    
    bbox_locations = [
        ("regular", Path("test") / "public"),
        ("normal_trimmed", Path("test") / "public" / "normal_trimmed")
    ]
    
    bbox_source_data = {
        'ped_anno': {},
        'veh_anno': {},
        'ped_gen': {},
        'veh_gen': {},
    }
    
    for location_name, path_suffix in bbox_locations:
        print(f"📦 Loading bbox data from {location_name} scenarios...")
        
        for bbox_type, category in [("ped_anno", "pedestrian"), ("veh_anno", "vehicle"), 
                                   ("ped_gen", "pedestrian"), ("veh_gen", "vehicle")]:
            if "anno" in bbox_type:
                base_dir = test_bbox_dir / "annotations" / "bbox_annotated"
            else:
                base_dir = test_bbox_dir / "annotations" / "bbox_generated"
                
            full_path = base_dir / category / path_suffix
            if full_path.exists():
                location_data = build_bbox_maps_fast(full_path, "test")
                bbox_source_data[bbox_type].update(location_data)
                print(f"  • {bbox_type} from {location_name}: {len(location_data)} camera stems")
    
    print(f"📊 Total bbox data loaded:")
    for bbox_type, data in bbox_source_data.items():
        print(f"  • {bbox_type}: {len(data)} camera stems")
    
    # Load external BDD_PC_5K bbox data from correct path
    external_bbox_dir = test_bbox_dir / "external" / "BDD_PC_5K"
    if external_bbox_dir.exists():
        print(f"📦 Loading external BDD_PC_5K bbox data from {external_bbox_dir}")
        external_bbox_data = {
            'combined_anno': build_bbox_maps_fast(external_bbox_dir / "annotations" / "bbox_annotated" / "test" / "public", "test"),
            'combined_gen': build_bbox_maps_fast(external_bbox_dir / "annotations" / "bbox_generated" / "test" / "public", "test"),
        }
        print(f"  • External combined_anno: {len(external_bbox_data['combined_anno'])} camera stems")
        print(f"  • External combined_gen: {len(external_bbox_data['combined_gen'])} camera stems")
    else:
        print(f"⚠️  External bbox directory not found: {external_bbox_dir}")
        external_bbox_data = {'combined_anno': {}, 'combined_gen': {}}

    # Find video directories
    test_public_dir = test_videos_dir / "videos" / "test" / "public"
    normal_trimmed_dir = test_public_dir / "normal_trimmed"
    external_videos_dir = test_videos_dir / "external"
    
    print(f"🎬 Checking video directories:")
    print(f"  • Normal trimmed: {normal_trimmed_dir} (exists: {normal_trimmed_dir.exists()})")
    print(f"  • External: {external_videos_dir} (exists: {external_videos_dir.exists()})")

    # Prepare tasks for processing
    all_tasks = []
    
    for i, vqa_entry in enumerate(vqa_test_data):
        scenario_videos = vqa_entry.get("videos", [])
        event_phases = vqa_entry.get("event_phase", [])
        environment_conversations = vqa_entry.get("conversations", [])
        
        # Must have videos and either environment questions OR event phases (or both)
        if not scenario_videos or (not event_phases and not environment_conversations):
            print(f"⚠️  Skipping incomplete VQA entry {i}: no videos or no questions")
            continue
        
        # Determine source and video directory based on video file name
        video_file = scenario_videos[0]
        
        if "normal_" in video_file:
            source = "main"
            video_base_dir = normal_trimmed_dir
            bbox_data = bbox_source_data
        elif video_file.startswith("video") and len(video_file.split("_")) == 1:
            source = "external"
            video_base_dir = external_videos_dir / "BDD_PC_5K" / "videos" / "test" / "public"
            bbox_data = external_bbox_data
        else:
            source = "main"
            video_base_dir = test_public_dir
            bbox_data = bbox_source_data
        
        task_data = (i, vqa_entry, video_base_dir, out_root, bbox_data, source)
        all_tasks.append(task_data)
    
    # Count different types of scenarios
    env_only_count = sum(1 for i, vqa_entry, _, _, _, _ in all_tasks 
                        if vqa_entry.get("conversations") and not vqa_entry.get("event_phase"))
    phase_only_count = sum(1 for i, vqa_entry, _, _, _, _ in all_tasks 
                          if vqa_entry.get("event_phase") and not vqa_entry.get("conversations"))
    both_count = sum(1 for i, vqa_entry, _, _, _, _ in all_tasks 
                    if vqa_entry.get("conversations") and vqa_entry.get("event_phase"))
    
    print(f"🚀 Processing {len(all_tasks)} VQA test tasks with best view selection...")
    print(f"   📊 Scenario breakdown:")
    print(f"      • Environment-only: {env_only_count} scenarios")
    print(f"      • Event-phase-only: {phase_only_count} scenarios") 
    print(f"      • Both types: {both_count} scenarios")
    all_samples = []
    
    with Pool(num_workers) as pool:
        results = pool.imap(process_test_scenario_task, all_tasks)
        
        for result in tqdm(results, total=len(all_tasks), ncols=100, 
                          desc=f"Processing VQA test (best view)"):
            if result is not None:
                all_samples.extend(result)
    
    # Write results
    print(f"📝 Writing {len(all_samples)} VQA test samples...")
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_jsonl, "w") as fout:
        for sample_data in all_samples:
            fout.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
    
    print(f"✅ Test processing complete: {len(all_samples)} VQA samples written to {out_jsonl}")
    return len(all_samples)

def find_pedestrian_bbox_for_view(view, bbox_source_data, source):
    """Find LARGEST pedestrian bbox area for a specific view across all phases - helper for environment view selection (ANNOTATED ONLY)"""
    try:
        largest_area = 0
        
        if source == "external":
            # External data: search through ONLY annotated bbox data (not generated)
            combined_anno_data = bbox_source_data.get('combined_anno', {})
            
            camera_stem = Path(view['camera_stem']).stem.replace(f"_f{view['frame_id']}", "")
            
            # Check ONLY annotated data for environment questions - find LARGEST across all phases
            if camera_stem in combined_anno_data:
                for phase_num, bbox_info in combined_anno_data[camera_stem].items():
                    if bbox_info is not None:
                        frame_id, bbox = bbox_info
                        if bbox and len(bbox) >= 4:
                            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            area = width * height
                            # Check minimum size requirements for external data
                            if width >= 50 and height >= 50 and area >= 3000:
                                if area > largest_area:
                                    largest_area = area
        else:
            # WTS scenarios: search through ONLY annotated pedestrian data (not generated)
            ped_anno_data = bbox_source_data.get('ped_anno', {})
            
            camera_stem = view['camera_stem'].replace(f"_f{view['frame_id']}", "")
            
            # Check ONLY annotated pedestrian data for environment questions - find LARGEST across all phases
            if camera_stem in ped_anno_data:
                for phase_num, bbox_info in ped_anno_data[camera_stem].items():
                    if bbox_info is not None:
                        frame_id, bbox = bbox_info
                        if bbox and len(bbox) >= 4:
                            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            area = width * height
                            # Check minimum size requirements for WTS data
                            if width >= 20 and height >= 20 and area >= 800:
                                if area > largest_area:
                                    largest_area = area
        
        return largest_area  # Return largest pedestrian bbox area found
        
    except Exception as e:
        print(f"      ⚠️  Error finding pedestrian bbox for view: {e}")
        return 0

def find_largest_pedestrian_bbox(scenario_videos, video_base_dir, bbox_source_data, source, original_scenario):
    """Find the largest pedestrian bbox across all views and phases for environment questions (ANNOTATED DATA ONLY)"""
    try:
        largest_pedestrian = None
        largest_area = 0
        
        print(f"      🔍 Searching through ALL frames for LARGEST pedestrian bbox...")
        
        if source == "external":
            # External data: search through ONLY annotated bbox data (not generated)
            combined_anno_data = bbox_source_data.get('combined_anno', {})
            
            for video_file in scenario_videos:
                camera_stem = Path(video_file).stem
                video_path = video_base_dir / video_file
                
                if not video_path.exists():
                    continue
                
                print(f"        📹 Checking camera: {camera_stem}")
                
                # Check ONLY annotated data for environment questions
                if camera_stem in combined_anno_data:
                    for phase_num, bbox_info in combined_anno_data[camera_stem].items():
                        if bbox_info is not None:
                            frame_id, bbox = bbox_info
                            if bbox and len(bbox) >= 4:
                                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                area = width * height
                                
                                print(f"          Phase {phase_num} Frame {frame_id}: {width}x{height} = {area} pixels")
                                
                                # Check minimum size requirements
                                if width >= 50 and height >= 50 and area >= 3000:
                                    if area > largest_area:
                                        print(f"          🎯 NEW LARGEST: {area} pixels (was {largest_area})")
                                        largest_area = area
                                        largest_pedestrian = {
                                            'video_path': video_path,
                                            'camera_stem': f"{camera_stem}_ped_annotated",
                                            'view_name': "single_view", 
                                            'frame_id': frame_id,
                                            'pedestrian_bbox': bbox,
                                            'pedestrian_bbox_area': area,
                                            'width': None,  # Will be set when extracting
                                            'height': None,
                                            'image_size': None
                                        }
                                    else:
                                        print(f"          ⚪ Smaller than current largest ({largest_area})")
                                else:
                                    print(f"          ❌ Too small (min: 50x50, 3000 area)")
        else:
            # WTS scenarios: search through ONLY annotated pedestrian data (not generated)
            ped_anno_data = bbox_source_data.get('ped_anno', {})
            
            for video_file in scenario_videos:
                scenario_name = extract_scenario_name(video_file)
                scenario_dir = video_base_dir / scenario_name
                
                if not scenario_dir.exists():
                    continue
                
                # Process ALL available camera views
                for camera_dir in scenario_dir.iterdir():
                    if not camera_dir.is_dir():
                        continue
                    
                    camera_view = camera_dir.name
                    if "overhead" in camera_view.lower():
                        view_type = "overhead_view"
                    elif "vehicle" in camera_view.lower():
                        view_type = "vehicle_view"
                    else:
                        view_type = "other_view"
                    
                    # Determine video path and camera stem
                    if "vehicle" in camera_view.lower():
                        scenario_name = extract_scenario_name(video_file)
                        vehicle_video_name = f"{scenario_name}_vehicle_view.mp4"
                        video_path = camera_dir / vehicle_video_name
                        camera_stem = f"{Path(video_file).stem}_vehicle_view"
                    else:
                        video_path = camera_dir / video_file
                        camera_stem = Path(video_file).stem
                    
                    if not video_path.exists():
                        continue
                    
                    print(f"        📹 Checking camera: {camera_stem} ({view_type})")
                    
                    # Check ONLY annotated pedestrian data for environment questions
                    if camera_stem in ped_anno_data:
                        for phase_num, bbox_info in ped_anno_data[camera_stem].items():
                            if bbox_info is not None:
                                frame_id, bbox = bbox_info
                                if bbox and len(bbox) >= 4:
                                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                    area = width * height
                                    
                                    print(f"          Phase {phase_num} Frame {frame_id}: {width}x{height} = {area} pixels")
                                    
                                    # Check minimum size requirements
                                    if width >= 20 and height >= 20 and area >= 800:
                                        if area > largest_area:
                                            print(f"          🎯 NEW LARGEST: {area} pixels (was {largest_area})")
                                            largest_area = area
                                            largest_pedestrian = {
                                                'video_path': video_path,
                                                'camera_stem': f"{camera_stem}_ped_annotated",
                                                'view_name': view_type,
                                                'frame_id': frame_id,
                                                'pedestrian_bbox': bbox,
                                                'pedestrian_bbox_area': area,
                                                'width': None,  # Will be set when extracting
                                                'height': None,
                                                'image_size': None
                                            }
                                        else:
                                            print(f"          ⚪ Smaller than current largest ({largest_area})")
                                    else:
                                        print(f"          ❌ Too small (min: 20x20, 800 area)")
        
        if largest_pedestrian:
            # Get actual frame dimensions
            try:
                from decord import VideoReader
                vr = VideoReader(str(largest_pedestrian['video_path']))
                frame_np = vr[largest_pedestrian['frame_id']].asnumpy()
                h, w = frame_np.shape[:2]
                largest_pedestrian['width'] = w
                largest_pedestrian['height'] = h
                largest_pedestrian['image_size'] = w * h
                print(f"      ✅ Found largest ANNOTATED pedestrian: {largest_area} pixels ({largest_pedestrian['view_name']})")
            except Exception as e:
                print(f"      ⚠️  Error getting frame dimensions: {e}")
                return None
        else:
            print(f"      ❌ No pedestrian bbox found meeting minimum requirements")
        
        return largest_pedestrian
        
    except Exception as e:
        print(f"      ❌ Error finding largest pedestrian bbox: {e}")
        return None

def process_environment_questions(scenario_id, vqa_entry, scenario_videos, video_base_dir, 
                                out_root, bbox_source_data, source, original_scenario, environment_conversations):
    """Process environment questions using the largest image + largest pedestrian bbox from the scenario"""
    try:
        print(f"    🔍 Finding largest image + largest pedestrian bbox for environment questions...")
        
        # Collect all available views and their image sizes
        candidate_views = []
        
        if source == "external":
            # External data: single video file
            for video_file in scenario_videos:
                video_path = video_base_dir / video_file
                if not video_path.exists():
                    continue
                    
                try:
                    from decord import VideoReader
                    vr = VideoReader(str(video_path))
                    
                    # Sample a few frames to get representative size (avoid frame 0)
                    sample_frames = [max(1, len(vr) // 6), len(vr) // 4, len(vr) // 2, 3 * len(vr) // 4, len(vr) - 1]
                    
                    for frame_idx in sample_frames:
                        frame_np = vr[frame_idx].asnumpy()
                        h, w = frame_np.shape[:2]
                        image_size = w * h
                        
                        candidate_views.append({
                            'video_path': video_path,
                            'camera_stem': f"{Path(video_file).stem}_f{frame_idx}",
                            'view_name': "single_view",
                            'frame_id': frame_idx,
                            'width': w,
                            'height': h,
                            'image_size': image_size
                        })
                        
                except Exception as e:
                    print(f"      ⚠️  Failed to process external video {video_file}: {e}")
                    continue
        
        else:
            # WTS scenarios with multiple camera views
            for video_file in scenario_videos:
                scenario_name = extract_scenario_name(video_file)
                scenario_dir = video_base_dir / scenario_name
                
                if not scenario_dir.exists():
                    continue
                
                # Process ALL available camera views
                for camera_dir in scenario_dir.iterdir():
                    if not camera_dir.is_dir():
                        continue
                    
                    camera_view = camera_dir.name
                    if "overhead" in camera_view.lower():
                        view_type = "overhead_view"
                    elif "vehicle" in camera_view.lower():
                        view_type = "vehicle_view"
                    else:
                        view_type = "other_view"
                    
                    # Determine video path and camera stem
                    if "vehicle" in camera_view.lower():
                        scenario_name = extract_scenario_name(video_file)
                        vehicle_video_name = f"{scenario_name}_vehicle_view.mp4"
                        video_path = camera_dir / vehicle_video_name
                        camera_stem = f"{Path(video_file).stem}_vehicle_view"
                    else:
                        video_path = camera_dir / video_file
                        camera_stem = Path(video_file).stem
                    
                    if not video_path.exists():
                        continue
                    
                    try:
                        from decord import VideoReader
                        vr = VideoReader(str(video_path))
                        
                        # Sample a few frames to get representative size (avoid frame 0)
                        sample_frames = [max(1, len(vr) // 6), len(vr) // 4, len(vr) // 2, 3 * len(vr) // 4, len(vr) - 1]
                        
                        for frame_idx in sample_frames:
                            frame_np = vr[frame_idx].asnumpy()
                            h, w = frame_np.shape[:2]
                            image_size = w * h
                            
                            candidate_views.append({
                                'video_path': video_path,
                                'camera_stem': f"{camera_stem}_f{frame_idx}",
                                'view_name': view_type,
                                'frame_id': frame_idx,
                                'width': w,
                                'height': h,
                                'image_size': image_size
                            })
                            
                    except Exception as e:
                        print(f"      ⚠️  Failed to process video {video_path}: {e}")
                        continue
        
        if not candidate_views:
            print(f"    ❌ No valid views found for environment questions")
            return []
        
        # PRIORITIZE VEHICLE_VIEW FIRST for environment questions, then by bbox area and image size
        vehicle_views = []
        other_views = []
        
        for view in candidate_views:
            if view['view_name'] == "vehicle_view":
                # Check if this vehicle view has pedestrian bbox data
                pedestrian_bbox_area = find_pedestrian_bbox_for_view(view, bbox_source_data, source)
                view['pedestrian_bbox_area'] = pedestrian_bbox_area
                vehicle_views.append(view)
            else:
                view['pedestrian_bbox_area'] = 0
                other_views.append(view)
        
        # Sort vehicle views by: 1) bbox area (largest first), 2) image_size (largest first)
        vehicle_views.sort(key=lambda x: (x['pedestrian_bbox_area'], x['image_size']), reverse=True)
        
        # Sort other views by: 1) view_type priority (overhead_view=5, others=1), 2) image_size (descending)
        def get_view_priority(view):
            view_type = view['view_name']
            if view_type == "overhead_view":
                type_priority = 5
            else:
                type_priority = 1
            return (type_priority, view['image_size'])
        
        other_views.sort(key=get_view_priority, reverse=True)
        
        # PRIORITIZE: VEHICLE_VIEW FIRST (by bbox area + size), then other views  
        candidate_views = vehicle_views + other_views
        
        # Select the 1 highest priority image (vehicle_view first, then largest)
        selected_views = []
        if candidate_views:
            selected_views.append(candidate_views[0])
        
        # Show the prioritization reasoning
        best_view = candidate_views[0] if candidate_views else None
        if best_view:
            if best_view['view_name'] == "vehicle_view":
                if best_view.get('pedestrian_bbox_area', 0) > 0:
                    print(f"    🎯 Selected 1 VEHICLE_VIEW (prioritized) with LARGEST PEDESTRIAN BBOX:")
                else:
                    print(f"    🎯 Selected 1 VEHICLE_VIEW (prioritized for env_largest):")
            else:
                print(f"    🎯 Selected 1 largest image (no vehicle_view available):")
        
        for i, view in enumerate(selected_views):
            bbox_info = f" (pedestrian bbox: {int(view.get('pedestrian_bbox_area', 0))} px²)" if view.get('pedestrian_bbox_area', 0) > 0 else ""
            view_priority_info = " [VEHICLE_VIEW PRIORITIZED]" if view['view_name'] == "vehicle_view" else ""
            print(f"      {i+1}. {view['camera_stem']} ({view['view_name']}) - {view['width']}x{view['height']} (size: {view['image_size']}){bbox_info}{view_priority_info}")
        
        # Now find the largest pedestrian bbox across all views and frames (ANNOTATED DATA ONLY)
        print(f"    🚶 Searching for largest ANNOTATED pedestrian bbox...")
        largest_pedestrian_view = find_largest_pedestrian_bbox(scenario_videos, video_base_dir, bbox_source_data, source, original_scenario)
        
        if largest_pedestrian_view:
            # Add the pedestrian view if it's different from the largest view
            if largest_pedestrian_view not in selected_views:
                selected_views.append(largest_pedestrian_view)
                print(f"    ✅ Added largest pedestrian view:")
                pedestrian_view_priority = " [VEHICLE_VIEW PRIORITIZED]" if largest_pedestrian_view['view_name'] == "vehicle_view" else ""
                print(f"      2. {largest_pedestrian_view['camera_stem']} ({largest_pedestrian_view['view_name']}) - Pedestrian bbox: {largest_pedestrian_view['pedestrian_bbox_area']}{pedestrian_view_priority}")
            else:
                print(f"    ℹ️  Largest pedestrian is in the same view as env_largest (VEHICLE_VIEW prioritized)")
        
        # Extract the selected images
        environment_images = []
        
        # Create env_largest image (vehicle view with largest bbox - full image)
        if selected_views:
            env_largest_view = selected_views[0]
            env_largest_rel_path = f"test/{original_scenario}/env_largest_{env_largest_view['camera_stem']}.jpg"
            env_largest_output_path = Path(out_root) / "images" / env_largest_rel_path
            
            # Extract and save the env_largest image (full size, no crop)
            W1, H1, _ = extract_and_save_frame_best_view(
                env_largest_view['video_path'],
                env_largest_view['frame_id'],
                env_largest_output_path,
                None,  # No bboxes for the largest view image
                None,
                env_largest_view['view_name'],
                source,
                should_crop=False  # Full image for env_largest
            )
            
            if W1 is not None:
                environment_images.append(str(env_largest_output_path.resolve()))
                print(f"      ✅ Extracted env_largest image: {W1}x{H1}")
            else:
                print(f"      ❌ Failed to extract env_largest image")
        
        # Create env_pedestrian image (best view with largest bbox - crop image)
        if largest_pedestrian_view:
            env_pedestrian_rel_path = f"test/{original_scenario}/env_pedestrian_{largest_pedestrian_view['camera_stem']}.jpg"
            env_pedestrian_output_path = Path(out_root) / "images" / env_pedestrian_rel_path
            
            # Extract and save the env_pedestrian image (crop only)
            W2, H2, _ = extract_and_save_frame_best_view(
                largest_pedestrian_view['video_path'],
                largest_pedestrian_view['frame_id'],
                env_pedestrian_output_path,
                largest_pedestrian_view.get('pedestrian_bbox'),  # Will draw green box if available
                None,  # No vehicle bbox for env_pedestrian
                largest_pedestrian_view['view_name'],
                source,
                should_crop=True  # Crop image for env_pedestrian
            )
            
            if W2 is not None:
                environment_images.append(str(env_pedestrian_output_path.resolve()))
                bbox_area = largest_pedestrian_view.get('pedestrian_bbox_area', 0)
                print(f"      ✅ Extracted env_pedestrian image: {W2}x{H2} (pedestrian area: {bbox_area})")
            else:
                print(f"      ❌ Failed to extract env_pedestrian image")
        
        if not environment_images:
            print(f"    ❌ No environment images extracted")
            return []
        
        # Create system prompt for environment questions
        env_system_prompt = """You are an expert traffic safety analyst specializing in environmental and contextual analysis of traffic scenes. 
Your task is to analyze traffic scene images and answer multiple-choice questions about environmental conditions, pedestrian characteristics, vehicle details, and scene context.

Visual Analysis Guidelines:
1. Focus on Environmental Details: Pay attention to weather conditions, lighting, road surface, traffic volume, and infrastructure
2. Pedestrian Analysis: Observe pedestrian appearance, clothing, age estimation, and physical characteristics. Look for pedestrians highlighted in green bounding boxes (some images may not have bounding boxes)
3. Vehicle Context: Note vehicle types, positions, and their interaction with the environment
4. Scene Context: Analyze road types, lane configurations, sidewalks, traffic patterns, and overall safety conditions

Question Format:
- Each question is multiple-choice with options A, B, C, and D (Note that: some questions may have fewer than 4 options)
- Choose the single best answer based on your visual analysis of the environment and scene context

Analysis Approach:
1. Examine the overall scene for environmental conditions (weather, lighting, road surface)
2. Identify infrastructure elements (road type, lanes, sidewalks, traffic signals)
3. Analyze pedestrian and vehicle characteristics visible in the scene
4. Select the most accurate answer based on evidence from the image"""
        
        # Create samples for each environment conversation
        environment_samples = []
        
        for i, conv in enumerate(environment_conversations):
            formatted_question = format_vqa_question(conv, "environment")
            
            # Create content array with environment images
            def build_content(text_content):
                content = []
                # Add environment images (largest + pedestrian bbox)
                for img_path in environment_images:
                    content.append({"type": "image", "url": img_path})
                # Add text content  
                content.append({"type": "text", "text": text_content})
                return content
            
            # Create conversation for environment question
            conversations_list = [
                {
                    "role": "system",
                    "content": env_system_prompt
                },
                {
                    "role": "user",
                    "content": build_content(formatted_question)
                }
            ]
            
            # Create sample for environment question
            sample_data = {
                "conversations": [conversations_list],
                "scenario": original_scenario,
                "phase_num": None,  # Environment questions are not phase-specific
                "phase_name": "environment",
                "question_id": conv.get("id", f"env_q_{i}"),
                "question_type": "environment",
                "phase_duration": None,
                "environment_info": {
                    "selection_type": "largest_image_plus_pedestrian_bbox",
                    "largest_view": {
                        "camera_stem": selected_views[0]['camera_stem'],
                        "view_name": selected_views[0]['view_name'],
                        "size": f"{selected_views[0]['width']}x{selected_views[0]['height']}",
                        "image_size": selected_views[0]['image_size']
                    } if selected_views else None,
                    "pedestrian_view": {
                        "camera_stem": selected_views[1]['camera_stem'],
                        "view_name": selected_views[1]['view_name'],
                        "size": f"{selected_views[1]['width']}x{selected_views[1]['height']}",
                        "pedestrian_bbox_area": selected_views[1]['pedestrian_bbox_area']
                    } if len(selected_views) > 1 and 'pedestrian_bbox' in selected_views[1] else None,
                    "num_images": len(environment_images),
                    "image_types": ["largest_view", "pedestrian_bbox"] if len(environment_images) > 1 else ["largest_view"]
                }
            }
            
            environment_samples.append(sample_data)
            print(f"      ✅ Created environment question {i+1}/{len(environment_conversations)}")
        
        return environment_samples
        
    except Exception as e:
        print(f"❌ Error processing environment questions for scenario {scenario_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_test_scenario_task(task_data):
    """Process a single test scenario with best view selection"""
    scenario_id, vqa_entry, video_base_dir, out_root, bbox_source_data, source = task_data
    
    try:
        scenario_videos = vqa_entry.get("videos", [])
        event_phases = vqa_entry.get("event_phase", [])
        environment_conversations = vqa_entry.get("conversations", [])  # Environment questions
        
        # Extract original scenario name from video filename
        if scenario_videos:
            video_file = scenario_videos[0]
            if source == "external":
                original_scenario = Path(video_file).stem
            else:
                original_scenario = extract_scenario_name(video_file)
        else:
            original_scenario = f"scenario_{scenario_id}"
        
        print(f"🎬 Processing VQA test scenario {scenario_id} ({original_scenario}) with BEST VIEW selection")
        print(f"  📊 Found {len(event_phases)} event phases and {len(environment_conversations)} environment questions")
        
        all_samples = []
        
        # First, process environment questions (use largest image + pedestrian bbox)
        if environment_conversations:
            print(f"  🌍 Processing {len(environment_conversations)} environment questions with largest image + pedestrian bbox")
            environment_samples = process_environment_questions(
                scenario_id, vqa_entry, scenario_videos, video_base_dir, 
                out_root, bbox_source_data, source, original_scenario, environment_conversations
            )
            if environment_samples:
                all_samples.extend(environment_samples)
                print(f"    ✅ Created {len(environment_samples)} environment samples")
        
        # Then process each event phase with best view selection
        for phase_idx, phase_data in enumerate(event_phases):
            phase_labels = phase_data.get("labels", [])
            conversations = phase_data.get("conversations", [])
            
            # Map phase label to phase number
            phase_name = phase_labels[0] if phase_labels else "unknown"
            phase_num = None
            for num, name in PHASE_ORDER.items():
                if name == phase_name:
                    phase_num = num
                    break
            
            if phase_num is None:
                print(f"    ⚠️  Unknown phase: {phase_name}")
                continue
            
            if not conversations:
                print(f"    ❌ No conversations found for phase {phase_num}")
                continue
            
            print(f"  📋 Processing Phase {phase_num} ({phase_name}) with {len(conversations)} questions")
            
            # Select the best views for this phase
            selected_views, has_bbox_data = select_best_views_for_phase(
                phase_num, scenario_videos, video_base_dir, 
                bbox_source_data, source, phase_data, original_scenario
            )
            
            if not selected_views:
                print(f"    ❌ No suitable views found for phase {phase_num}")
                continue
            
            # Extract frames based on bbox availability
            phase_images = []
            
            if has_bbox_data:
                # Single view with bbox - generate crop + full
                view = selected_views[0]
                frame_choice = view['frame_choice']
                
                # Create output paths for both full and cropped versions
                full_rel_path = f"test/{original_scenario}/best_view_{view['camera_stem']}_phase{phase_num}_f{frame_choice['frame']}_full.jpg"
                crop_rel_path = f"test/{original_scenario}/best_view_{view['camera_stem']}_phase{phase_num}_f{frame_choice['frame']}_crop.jpg"
                
                full_output_path = Path(out_root) / "images" / full_rel_path
                crop_output_path = Path(out_root) / "images" / crop_rel_path
                
                # Extract and save full version (no resize)
                W_full, H_full, _ = extract_and_save_frame_best_view(
                    view['video_path'],
                    frame_choice['frame'],
                    full_output_path,
                    frame_choice['ped_box'], 
                    frame_choice['veh_box'],
                    view['view_name'],
                    source,
                    should_crop=False
                )
                
                # Extract and save cropped version
                W_crop, H_crop, has_crop = extract_and_save_frame_best_view(
                    view['video_path'],
                    frame_choice['frame'],
                    crop_output_path,
                    frame_choice['ped_box'], 
                    frame_choice['veh_box'],
                    view['view_name'],
                    source,
                    should_crop=True
                )
                
                # Add both images if successfully extracted
                if W_full is not None:
                    phase_images.append(str(full_output_path.resolve()))
                if W_crop is not None and has_crop:
                    phase_images.append(str(crop_output_path.resolve()))
                
                if not phase_images:
                    print(f"    ❌ Failed to extract images for phase {phase_num}")
                    continue
                
                print(f"    ✅ Extracted BBOX view: FULL + CROP from {view['camera_stem']}")
                
            else:
                # Multiple views without bbox - generate 2 full versions (no resize, no crop)
                for i, view in enumerate(selected_views):
                    frame_choice = view['frame_choice']
                    
                    # Create output path for full version
                    full_rel_path = f"test/{original_scenario}/best_view_{view['camera_stem']}_phase{phase_num}_f{frame_choice['frame']}_view{i+1}.jpg"
                    full_output_path = Path(out_root) / "images" / full_rel_path
                    
                    # Extract and save full version (no resize)
                    W, H, _ = extract_and_save_frame_best_view(
                        view['video_path'],
                        frame_choice['frame'],
                        full_output_path,
                        frame_choice['ped_box'], 
                        frame_choice['veh_box'],
                        view['view_name'],
                        source,
                        should_crop=False
                    )
                    
                    if W is None:
                        print(f"    ❌ Failed to extract view {i+1} for phase {phase_num}")
                        continue
                    
                    # Add the image
                    phase_images.append(str(full_output_path.resolve()))
                    
                print(f"    ✅ Extracted NO-BBOX views: 2 FULL versions from {len(selected_views)} cameras")
                
            if len(phase_images) == 0:
                print(f"    ❌ No images extracted for phase {phase_num}")
                continue
            
            # Create system prompt for VQA task
            system_prompt = create_vqa_system_prompt(phase_name, PHASE_DESCRIPTIONS[phase_name])
            
            print(f"    ✅ Best view(s) extracted with {len(conversations)} questions")
            
            # Create individual samples for each question (using extracted images)
            for i, conv in enumerate(conversations):
                formatted_question = format_vqa_question(conv, "best_view")
                
                # Create content array with extracted images (max 2)
                def build_content(text_content):
                    content = []
                    # Add extracted images (already max 2)
                    for img_path in phase_images:
                        content.append({"type": "image", "url": img_path})
                    # Add text content  
                    content.append({"type": "text", "text": text_content})
                    return content
                
                # Create conversation in new format with system message
                conversations_list = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": build_content(formatted_question)
                    }
                ]
                
                # Create sample for this question
                if has_bbox_data:
                    # Single view with bbox data
                    view_info = {
                        "selection_type": "single_view_with_bbox",
                        "camera_stem": selected_views[0]['camera_stem'],
                        "view_name": selected_views[0]['view_name'],
                        "priority": selected_views[0]['priority'],
                        "has_bbox": True,
                        "num_images": 2,
                        "image_types": ["full", "crop"]
                    }
                else:
                    # Multiple views without bbox
                    view_info = {
                        "selection_type": "multi_view_no_bbox",
                        "views": [{"camera_stem": v['camera_stem'], "view_name": v['view_name'], "priority": v['priority']} for v in selected_views],
                        "has_bbox": False,
                        "num_images": len(phase_images),
                        "image_types": ["full"] * len(phase_images)
                    }
                
                sample_data = {
                    "conversations": [conversations_list],
                    "scenario": original_scenario,
                    "phase_num": phase_num,
                    "phase_name": phase_name,
                    "question_id": conv.get("id", f"q_{i}"),
                    "question_type": "vqa",
                    "phase_duration": float(phase_data.get("end_time", 0)) - float(phase_data.get("start_time", 0)),
                    "best_view_info": view_info
                }
                
                all_samples.append(sample_data)
                
                image_type_desc = "full + crop" if has_bbox_data else f"{len(phase_images)} full views"
                print(f"      ✅ Created question {i+1}/{len(conversations)} with {image_type_desc}")
        
        return all_samples if all_samples else None
        
    except Exception as e:
        print(f"❌ Error processing test scenario {scenario_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_bbox_maps_fast(bbox_root, split):
    """Build bbox mappings for fast lookup (same as original)"""
    bbox_root = Path(bbox_root)
    if not bbox_root.exists():
        print(f"⚠️  Bbox directory not found: {bbox_root}")
        return {}
    
    json_files = list(bbox_root.rglob("*.json"))
    if not json_files:
        print(f"⚠️  No JSON files found in: {bbox_root}")
        return {}
    
    print(f"📦 Loading {len(json_files)} bbox files from {bbox_root}")
    print(f"   Sample files: {[f.name for f in json_files[:5]]}")
    
    mp = {}
    with Pool(min(16, len(json_files))) as pool:
        results = pool.map(process_bbox_file, json_files)
    
    for stem, bbox_data in results:
        if bbox_data:
            if stem not in mp:
                mp[stem] = {}
            mp[stem].update(bbox_data)
    
    print(f"   Processed {len(mp)} camera stems with bbox data")
    if mp:
        print(f"   Sample camera stems: {list(mp.keys())[:3]}")
        first_stem = list(mp.keys())[0]
        first_data = mp[first_stem]
        print(f"   Sample data for '{first_stem}': {list(first_data.keys())} phases")
    
    return mp

def process_bbox_file(js):
    """Helper for parallel bbox file processing"""
    stem = js.stem.replace("_bbox", "")
    return stem, coco2map_fast(js)

def extract_scenario_name(video_file):
    """Extract base scenario name from video file name"""
    video_name = Path(video_file).stem
    
    if "normal_" in video_name:
        return video_name
    
    import re
    match = re.match(r'(\d{8}_\d+_[A-Z]+\d*_T\d+)', video_name)
    if match:
        return match.group(1)
    
    return video_name

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--test_videos_dir", required=True, help="Test videos directory (WTS_DATASET_PUBLIC_TEST)")
    pa.add_argument("--test_bbox_dir", required=True, help="Test bbox directory (WTS_DATASET_PUBLIC_TEST_BBOX)")  
    pa.add_argument("--vqa_test_file", required=True, help="VQA test file (WTS_VQA_PUBLIC_TEST.json)")
    pa.add_argument("--out_root", required=True, help="Output directory for processed images")
    pa.add_argument("--out_jsonl", required=True, help="Output JSONL file path")
    pa.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers (default: 32)")
    args = pa.parse_args()

    print("🚀 WTS Dataset VQA Test Processor - SUBTASK 2 WITH BEST VIEW SELECTION")
    print("="*70)
    print("📋 Configuration:")
    print(f"  • Test videos: {args.test_videos_dir}")
    print(f"  • Test bbox: {args.test_bbox_dir}")
    print(f"  • VQA test file: {args.vqa_test_file}")
    print(f"  • Output root: {args.out_root}")
    print(f"  • Output JSONL: {args.out_jsonl}")
    print(f"  • Parallel workers: {args.num_workers}")
    print("📊 Data Sources:")
    print(f"  • Normal trimmed videos: videos/test/public/normal_trimmed/")
    print(f"  • External videos: external/")
    print(f"  • Test VQA questions: {Path(args.vqa_test_file).name}")
    print("🎯 Features:")
    print(f"  • BEST VIEW SELECTION: Prioritizes views with bbox annotations for phase questions")
    print(f"  • ENVIRONMENT QUESTIONS: Uses 1 largest image from scenario for environment analysis")
    print(f"  • Dual output: Both full and cropped versions for phase questions with bbox")
    print(f"  • Multi-view output: 2 full versions for phase questions without bbox")
    print(f"  • Intelligent view ranking based on annotation quality and view type")
    print(f"  • Test format: questions without ground truth answers")
    print("="*70)

    overall_start = time.time()
    
    total_samples = process_test_data_fast(
        args.test_videos_dir, 
        args.test_bbox_dir,
        args.vqa_test_file,
        args.out_root, 
        args.out_jsonl, 
        args.num_workers
    )
    
    total_time = time.time() - overall_start
    
    print(f"\n🎉 VQA TEST PROCESSING COMPLETE!")
    print(f"📊 FINAL PERFORMANCE:")
    print(f"  • Total VQA test samples: {total_samples}")
    print(f"  • Total time: {total_time:.2f}s")
    print(f"  • Overall speed: {total_samples/total_time:.2f} samples/sec")
    print(f"🔍 BEST VIEW TEST VQA DATASET CREATED: {args.out_jsonl}")

if __name__ == "__main__":
    main() 