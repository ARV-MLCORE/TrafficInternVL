#!/usr/bin/env python3
# prepare_data_train_subtask2_best_view.py - VQA TRAINING WITH BEST VIEW SELECTION + ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
# • Adapted from subtask2 for best view selection approach in training data
# • Processes both phase-based questions AND environment questions
# • Phase questions: Selects best view per phase (prioritizing bbox annotations)
# • Environment questions: Prioritizes vehicle view + largest images, avoids frame 0  
# • Max 2 images per sample: [crop, full] when bbox available or [full] when no bbox
# • No image resizing - preserves original resolution for better VLM training
# • Uses intelligent view selection based on bbox availability and view quality
# • Updated JSON format with system field at same level as conversations
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

# ---------- 2. Helper functions (adapted from best_view approach) --------------------
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
    """No resizing - return image as-is"""
    return image

def extract_and_save_frame_best_view(video_path, frame_id, output_path, box_p=None, box_v=None, view_name=None, source="main", should_crop=True, save_full_too=True):
    """Enhanced frame extraction - saves [crop, full] when bbox available, or [full] only."""
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
        
        saved_paths = []
        
        # Strategy: Save CROP if bbox exists and should_crop is True
        if has_bbox and should_crop:
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
                
                # Save cropped version with "_crop" suffix
                crop_output_path = str(output_path).replace('.jpg', '_crop.jpg')
                if cropped_frame.size > 0 and cropped_frame.shape[0] > 10 and cropped_frame.shape[1] > 10:
                    cv2.imwrite(crop_output_path, cropped_frame)  # No resizing
                    saved_paths.append(str(Path(crop_output_path).resolve()))
                    print(f"      SAVED: Saved cropped: {Path(crop_output_path).name} ({cropped_frame.shape[1]}x{cropped_frame.shape[0]})")
                else:
                    print(f"      WARNING:  Invalid crop, will save only full frame")
        
        # Strategy: Save FULL frame if conditions are met
        # - Always save full frame if it's requested (save_full_too=True)
        # - OR if no bbox was present to begin with
        if save_full_too or not has_bbox:
            cv2.imwrite(str(output_path), frame)  # No resizing
            saved_paths.append(str(Path(output_path).resolve()))
            print(f"      SAVED: Saved full: {Path(output_path).name} ({frame.shape[1]}x{frame.shape[0]})")
        
        # Ensure we have max 2 images: [crop] or [full] or [crop, full] when both are needed
        if len(saved_paths) > 2:
            print(f"      WARNING:  Too many images ({len(saved_paths)}), keeping first 2")
            saved_paths = saved_paths[:2]
        
        print(f"      SUCCESS: Total images for this sample: {len(saved_paths)} (max 2)")
        return W, H, saved_paths
        
    except Exception as e:
        logging.warning(f"Failed to extract frame {frame_id} from {video_path}: {e}")
        return None, None, []

def select_best_view_for_phase(phase_num, view_files, videos_dir, bbox_data, source, scenario):
    """Select the best view for a phase based on bbox availability and view priority"""
    
    candidate_views = []
    
    # Collect all available views with their priorities
    for view_name, vqa_fp in view_files.items():
        try:
            if source == "external":
                # External data: single video file approach
                video_path = videos_dir / f"{scenario}.mp4"
                if not video_path.exists():
                    continue
                    
                camera_stem = scenario
                
                # Get annotated frame for this phase (external data)
                frame_choice = get_annotated_frame_for_phase(phase_num, camera_stem, bbox_data, source)
                
                if frame_choice:
                    # Calculate priority: bbox availability + view type priority
                    bbox_priority = 10 if (frame_choice.get('ped_box') or frame_choice.get('veh_box')) else 0
                    view_priority = VIEW_PRIORITY.get(view_name, 0)
                    total_priority = bbox_priority + view_priority
                    
                    candidate_views.append({
                        'video_path': video_path,
                        'camera_stem': camera_stem,
                        'view_name': view_name,
                        'frame_choice': frame_choice,
                        'priority': total_priority,
                        'has_bbox': bbox_priority > 0
                    })
        except Exception as e:
            logging.warning(f"Error processing view {view_name}: {e}")
    
    # Select the best view based on priority
    if candidate_views:
        # Sort by priority (highest first)
        candidate_views.sort(key=lambda x: x['priority'], reverse=True)
        best_view = candidate_views[0]
        print(f"    TARGET: Selected best view: {best_view['camera_stem']} ({best_view['view_name']}) - Priority: {best_view['priority']}")
        return best_view
    
    return None

def get_annotated_frame_for_phase(phase_num, camera_stem, bbox_data, source="main"):
    """Get the annotated frame for a specific phase and camera"""
    
    if source == "external":
        # External BDD_PC_5K has combined annotations (one bbox per phase)
        combined_data = bbox_data.get('combined_anno', {}).get(camera_stem, {})
        if phase_num in combined_data:
            frame_id, bbox = combined_data[phase_num]
            # For external data, always treat as pedestrian (GREEN box)
            return {
                'frame': frame_id,
                'ped_box': bbox,  # Always treat external bboxes as pedestrian
                'veh_box': None,
                'has_annotation': True
            }
        return None
    else:
        # Main WTS dataset with separate pedestrian and vehicle annotations
        # Get phase-based annotations for this specific camera
        ped_phase_data = bbox_data['ped_anno'].get(camera_stem, {})
        veh_phase_data = bbox_data['veh_anno'].get(camera_stem, {})
        
        # Check if this phase has annotations
        ped_annotation = ped_phase_data.get(phase_num)  # (frame_id, bbox)
        veh_annotation = veh_phase_data.get(phase_num)  # (frame_id, bbox)
        
        # In WTS dataset, each phase should have consistent frame_id across ped/veh annotations
        if ped_annotation or veh_annotation:
            # Get frame_id (should be same for both ped and veh if both exist)
            frame_id = None
            ped_box = None
            veh_box = None
            
            if ped_annotation:
                frame_id, ped_box = ped_annotation
            if veh_annotation:
                frame_id_veh, veh_box = veh_annotation
                if frame_id is None:
                    frame_id = frame_id_veh
                elif frame_id != frame_id_veh:
                    print(f"    WARNING:  Frame ID mismatch for phase {phase_num} camera {camera_stem}: ped={frame_id}, veh={frame_id_veh}")
                    # Use pedestrian frame_id as primary
            
            if frame_id is not None:
                return {
                    'frame': frame_id,
                    'ped_box': ped_box,
                    'veh_box': veh_box,
                    'has_annotation': True
                }
        
        # No annotations found for this phase and camera
        return None

def select_best_view_for_environment(view_files, videos_dir, bbox_data, source, scenario):
    """Select the best view for environment questions - prioritize vehicle view with largest pedestrian bbox, then largest images"""
    
    candidate_views = []
    
    # Collect all available views with their priorities
    for view_name, vqa_fp in view_files.items():
        try:
            if source == "external":
                # External data: single video file approach
                video_path = videos_dir / f"{scenario}.mp4"
                if not video_path.exists():
                    continue
                    
                camera_stem = scenario
                
                # For environment questions, we want to find the largest/best frame
                # Check multiple frames but avoid frame 0
                best_frame = find_best_environment_frame(video_path, camera_stem, bbox_data, source, avoid_frame_0=True)
                
                if best_frame:
                    # Calculate priority: vehicle view gets highest priority, then by view quality
                    base_priority = 20 if view_name == "vehicle_view" else VIEW_PRIORITY.get(view_name, 0)
                    
                    # For vehicle view, add pedestrian bbox size bonus
                    pedestrian_bbox_bonus = 0
                    if view_name == "vehicle_view" and best_frame.get('ped_box'):
                        ped_box = best_frame['ped_box']
                        if ped_box and len(ped_box) >= 4:
                            bbox_area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                            # Normalize bbox area bonus (0-10 points based on area)
                            pedestrian_bbox_bonus = min(bbox_area / 10000, 10)  # Scale bbox area to 0-10 points
                    
                    total_priority = base_priority + pedestrian_bbox_bonus
                    
                    candidate_views.append({
                        'video_path': video_path,
                        'camera_stem': camera_stem,
                        'view_name': view_name,
                        'frame_choice': best_frame,
                        'priority': total_priority,
                        'is_vehicle_view': view_name == "vehicle_view",
                        'pedestrian_bbox_area': pedestrian_bbox_bonus * 1000 if pedestrian_bbox_bonus > 0 else 0
                    })
            else:
                # Main WTS dataset with nested structure
                video_dir = videos_dir / scenario / view_name
                if not video_dir.exists():
                    continue
                    
                vids = sorted(video_dir.glob("*.mp4"))
                if not vids:
                    continue
                
                # Process each camera in this view
                for vid in vids:
                    camera_stem = vid.stem
                    
                    # For environment questions, find the best frame from this camera
                    best_frame = find_best_environment_frame(vid, camera_stem, bbox_data, source, avoid_frame_0=True)
                    
                    if best_frame:
                        # Calculate priority: vehicle view gets highest priority, then by view quality
                        base_priority = 20 if view_name == "vehicle_view" else VIEW_PRIORITY.get(view_name, 0)
                        
                        # For vehicle view, add pedestrian bbox size bonus
                        pedestrian_bbox_bonus = 0
                        if view_name == "vehicle_view" and best_frame.get('ped_box'):
                            ped_box = best_frame['ped_box']
                            if ped_box and len(ped_box) >= 4:
                                bbox_area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                                # Normalize bbox area bonus (0-10 points based on area)
                                pedestrian_bbox_bonus = min(bbox_area / 10000, 10)  # Scale bbox area to 0-10 points
                        
                        total_priority = base_priority + pedestrian_bbox_bonus
                        
                        candidate_views.append({
                            'video_path': vid,
                            'camera_stem': camera_stem,
                            'view_name': view_name,
                            'frame_choice': best_frame,
                            'priority': total_priority,
                            'is_vehicle_view': view_name == "vehicle_view",
                            'pedestrian_bbox_area': pedestrian_bbox_bonus * 1000 if pedestrian_bbox_bonus > 0 else 0
                        })
        except Exception as e:
            print(f"    WARNING:  Error processing {view_name} for environment: {e}")
            continue
    
    # Select the best view based on priority: VEHICLE_VIEW FIRST, then by priority and image size
    if candidate_views:
        # Separate vehicle views from other views
        vehicle_views = [v for v in candidate_views if v['view_name'] == "vehicle_view"]
        other_views = [v for v in candidate_views if v['view_name'] != "vehicle_view"]
        
        # Sort vehicle views by priority and image size
        vehicle_views.sort(key=lambda x: (x['priority'], x['frame_choice'].get('image_size', 0)), reverse=True)
        
        # Sort other views by priority and image size
        other_views.sort(key=lambda x: (x['priority'], x['frame_choice'].get('image_size', 0)), reverse=True)
        
        # Prioritize: vehicle_view first, then other views
        prioritized_views = vehicle_views + other_views
        best_view = prioritized_views[0]
        
        view_type_info = "VEHICLE_VIEW (prioritized)" if best_view['view_name'] == "vehicle_view" else f"{best_view['view_name'].upper()}"
        bbox_info = f" (pedestrian bbox: {int(best_view['pedestrian_bbox_area'])} px²)" if best_view['pedestrian_bbox_area'] > 0 else ""
        print(f"    TARGET: Selected best environment view: {best_view['camera_stem']} ({view_type_info}) - Priority: {best_view['priority']:.1f}{bbox_info}")
        return best_view
    
    return None

def find_best_environment_frame(video_path, camera_stem, bbox_data, source="main", avoid_frame_0=True):
    """Find the best frame for environment questions - prioritize frames with LARGEST PEDESTRIAN BBOXES"""
    
    try:
        from decord import VideoReader
        vr = VideoReader(str(video_path))
        total_frames = len(vr)
        
        if total_frames == 0:
            return None
        
        # Get annotation data for this camera
        if source == "external":
            # External data: check ONLY annotated data (not generated) for environment questions
            combined_data = bbox_data.get('combined_anno', {}).get(camera_stem, {})
            
            # For environment questions, prioritize LARGEST PEDESTRIAN BBOX across all frames
            best_frame = None
            largest_ped_area = 0
            
            for phase_num, (frame_id, bbox) in combined_data.items():
                if avoid_frame_0 and frame_id <= 0:
                    continue
                if frame_id >= total_frames:
                    continue
                    
                if bbox and len(bbox) >= 4:
                    # Calculate pedestrian bbox area
                    ped_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    # Check minimum size requirements for external data
                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if width >= 50 and height >= 50 and ped_area >= 3000:
                        
                        # Prioritize LARGEST pedestrian bbox for environment questions
                        if ped_area > largest_ped_area:
                            largest_ped_area = ped_area
                            
                            # Get frame size for additional context
                            frame_np = vr[frame_id].asnumpy()
                            frame_size = frame_np.shape[0] * frame_np.shape[1]
                            
                            best_frame = {
                                'frame': frame_id,
                                'ped_box': bbox,  # Always treat external as pedestrian
                                'veh_box': None,
                                'has_annotation': True,
                                'image_size': frame_size,
                                'phase_num': phase_num,
                                'bbox_area': ped_area
                            }
                            print(f"      TARGET: Found larger pedestrian bbox: {ped_area} pixels in phase {phase_num} frame {frame_id}")
            
            if best_frame:
                print(f"      SUCCESS: Selected frame with LARGEST pedestrian bbox: {largest_ped_area} pixels")
                return best_frame
                
        else:
            # Main WTS dataset: check ONLY annotated pedestrian data (not generated) for environment questions
            ped_data = bbox_data['ped_anno'].get(camera_stem, {})
            veh_data = bbox_data['veh_anno'].get(camera_stem, {})
            
            # For environment questions, prioritize LARGEST PEDESTRIAN BBOX across all frames
            best_frame = None
            largest_ped_area = 0
            
            for phase_num in range(5):
                if phase_num not in ped_data:
                    continue
                    
                frame_id, ped_bbox = ped_data[phase_num]
                
                if avoid_frame_0 and frame_id <= 0:
                    continue
                if frame_id >= total_frames:
                    continue
                    
                if ped_bbox and len(ped_bbox) >= 4:
                    # Calculate pedestrian bbox area
                    ped_area = (ped_bbox[2] - ped_bbox[0]) * (ped_bbox[3] - ped_bbox[1])
                    
                    # Check minimum size requirements for WTS data
                    width, height = ped_bbox[2] - ped_bbox[0], ped_bbox[3] - ped_bbox[1]
                    if width >= 20 and height >= 20 and ped_area >= 800:
                        
                        # Prioritize LARGEST pedestrian bbox for environment questions
                        if ped_area > largest_ped_area:
                            largest_ped_area = ped_area
                            
                            # Get frame size for additional context
                            frame_np = vr[frame_id].asnumpy()
                            frame_size = frame_np.shape[0] * frame_np.shape[1]
                            
                            # Get vehicle bbox if available for the same frame
                            veh_bbox = None
                            if phase_num in veh_data:
                                veh_frame_id, veh_bbox = veh_data[phase_num]
                                if veh_frame_id != frame_id:
                                    veh_bbox = None  # Only use if same frame
                            
                            best_frame = {
                                'frame': frame_id,
                                'ped_box': ped_bbox,
                                'veh_box': veh_bbox,
                                'has_annotation': True,
                                'image_size': frame_size,
                                'phase_num': phase_num,
                                'bbox_area': ped_area
                            }
                            print(f"      TARGET: Found larger pedestrian bbox: {ped_area} pixels in phase {phase_num} frame {frame_id}")
            
            if best_frame:
                print(f"      SUCCESS: Selected frame with LARGEST pedestrian bbox: {largest_ped_area} pixels")
                return best_frame
        
        # No annotated frames available, pick a good frame from the middle of the video
        # Avoid frame 0 and very last frames
        start_frame = 1 if avoid_frame_0 else 0
        end_frame = max(start_frame + 1, total_frames - 5)
        middle_frame = min(total_frames // 2, end_frame)
        
        # Ensure we don't go out of bounds
        selected_frame = max(start_frame, min(middle_frame, total_frames - 1))
        
        # Get frame size
        frame_np = vr[selected_frame].asnumpy()
        frame_size = frame_np.shape[0] * frame_np.shape[1]
        
        return {
            'frame': selected_frame,
            'ped_box': None,
            'veh_box': None,
            'has_annotation': False,
            'image_size': frame_size,
            'phase_num': None
        }
        
    except Exception as e:
        print(f"    WARNING:  Error finding best environment frame for {camera_stem}: {e}")
        return None

def load_environment_questions(scenario_dir, scenario, source="main"):
    """Load environment questions from the environment directory"""
    try:
        if source == "external":
            # External BDD_PC_5K: look for environment subdirectory
            env_file = scenario_dir / "environment" / f"{scenario}.json"
        else:
            # Main WTS dataset: environment directory in scenario
            env_file = scenario_dir / "environment" / f"{scenario}.json"
        
        if not env_file.exists():
            return None
        
        with open(env_file, 'r') as f:
            env_data = json.load(f)
        
        # Handle different environment JSON structures
        if isinstance(env_data, list) and len(env_data) > 0:
            # Main WTS format: list with single object
            env_questions = env_data[0].get("environment", [])
        elif isinstance(env_data, dict):
            # Alternative format: direct object
            env_questions = env_data.get("environment", [])
        else:
            return None
        
        if not env_questions:
            return None
        
        print(f"    INFO: Loaded {len(env_questions)} environment questions for {scenario}")
        return env_questions
        
    except Exception as e:
        print(f"    WARNING:  Failed to load environment questions for {scenario}: {e}")
        return None

def create_environment_system_prompt():
    """Create a system prompt for environment questions"""
    return """You are an expert traffic safety analyst specializing in environmental and contextual analysis of traffic scenes. 
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

# ---------- 3. VQA-specific functions ------------------------------------------------

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

2. Best View Analysis: The image may contain the best available view (cropped or full) selected for optimal VLM guidance. Analyze all visible elements carefully.

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
    """Format a VQA question with multiple choice options and proper answer format"""
    question = question_data["question"]
    choices = []
    choice_mapping = {}
    
    # Collect available choices (a, b, c, d)
    for choice_key in ['a', 'b', 'c', 'd']:
        if choice_key in question_data and question_data[choice_key].strip():
            choice_letter = choice_key.upper()
            choice_text = question_data[choice_key]
            choices.append(f"{choice_letter}. {choice_text}")
            choice_mapping[choice_key] = f"{choice_letter}. {choice_text}"
    
    # Get correct answer
    correct_key = question_data.get("correct", "").lower()
    if correct_key in choice_mapping:
        correct_answer = choice_mapping[correct_key]
    else:
        # Fallback: if correct key not found, use the raw correct value
        correct_answer = question_data.get("correct", "")
        if correct_answer and not correct_answer.startswith(('A', 'B', 'C', 'D')):
            # If correct answer doesn't start with letter, try to map it
            for key, value in choice_mapping.items():
                if correct_answer.lower() in value.lower():
                    correct_answer = value
                    break
    
    # Format the question with choices
    if choices:
        choices_text = "\n".join(choices)
        formatted_question = f"{question}\n{choices_text}"
    else:
        formatted_question = question
    
    return formatted_question, correct_answer

def process_split_fast(data_root, out_root, out_jsonl, split, num_workers=32):
    """Main processing function adapted for VQA data with best view selection"""
    data_root = Path(data_root)
    
    # VQA annotation directories
    vqa_dir = data_root / "annotations" / "vqa" / split
    vqa_normal_trimmed_dir = data_root / "annotations" / "vqa" / split / "normal_trimmed"
    external_bdd_dir = data_root / "external" / "BDD_PC_5K"
    
    # Video directories (same as subtask 1)
    videos_dir = data_root / "videos" / split
    normal_trimmed_videos_dir = videos_dir / "normal_trimmed"
    
    # Bbox annotation directories (same as subtask 1)
    bbox_anno_dir = data_root / "annotations" / "bbox_annotated"
    bbox_gen_dir = data_root / "annotations" / "bbox_generated"

    # Verify directories
    if not vqa_dir.exists() or not videos_dir.exists():
        print(f"ERROR: Required directories missing for {split}")
        print(f"   VQA dir: {vqa_dir} (exists: {vqa_dir.exists()})")
        print(f"   Videos dir: {videos_dir} (exists: {videos_dir.exists()})")
        return 0

    # Pre-load all bbox annotation data (same as subtask 1)
    print(f"LOADING: Pre-loading {split} bbox annotation data...")
    start_time = time.time()
    
    bbox_data = {
        'ped_anno': build_bbox_maps_fast(bbox_anno_dir / "pedestrian" / split, split),
        'veh_anno': build_bbox_maps_fast(bbox_anno_dir / "vehicle" / split, split),
        'ped_gen': build_bbox_maps_fast(bbox_gen_dir / "pedestrian" / split, split),
        'veh_gen': build_bbox_maps_fast(bbox_gen_dir / "vehicle" / split, split),
    }
    
    # Load external BDD_PC_5K annotations if available
    external_bbox_data = {}
    if external_bdd_dir.exists():
        print(f"LOADING: Loading external BDD_PC_5K bbox annotations for {split}...")
        external_annotations = external_bdd_dir / "annotations"
        if external_annotations.exists():
            external_bbox_data = {
                'combined_anno': build_bbox_maps_fast(external_annotations / "bbox_annotated" / split, split),
                'combined_gen': build_bbox_maps_fast(external_annotations / "bbox_generated" / split, split),
            }
    
    load_time = time.time() - start_time
    print(f"FAST: {split} bbox data loaded in {load_time:.2f}s")

    # Find all VQA files
    vqa_paths = list(vqa_dir.rglob("*.json"))
    
    # Add normal_trimmed VQA files if they exist
    if vqa_normal_trimmed_dir.exists():
        vqa_normal_trimmed_paths = list(vqa_normal_trimmed_dir.rglob("*.json"))
        vqa_paths.extend(vqa_normal_trimmed_paths)
    
    # Also check external BDD_PC_5K VQA files
    external_vqa_paths = []
    if external_bdd_dir.exists():
        external_vqa_dir = external_bdd_dir / "annotations" / "vqa" / split
        print(f"SEARCHING: Checking external VQA dir: {external_vqa_dir}")
        if external_vqa_dir.exists():
            external_vqa_paths = list(external_vqa_dir.rglob("*.json"))
            print(f"SEARCHING: Found {len(external_vqa_paths)} external VQA files")
        else:
            print(f"WARNING:  External VQA directory not found: {external_vqa_dir}")
    else:
        print(f"WARNING:  External BDD directory not found: {external_bdd_dir}")

    all_vqa_paths = vqa_paths + external_vqa_paths
    print(f"SEARCHING: Total VQA files found: {len(vqa_paths)} main + {len(external_vqa_paths)} external = {len(all_vqa_paths)} for {split}")

    if len(all_vqa_paths) == 0:
        print(f"ERROR: No VQA files found for {split}")
        return 0

    # Group VQA files by scenario
    scenarios = {}
    external_scenarios = {}
    
    # Process main WTS dataset scenarios
    for vqa_fp in vqa_paths:
        view_dir = vqa_fp.parent
        scenario = view_dir.parent.name
        view_name = view_dir.name
        
        if scenario not in scenarios:
            scenarios[scenario] = {}
        scenarios[scenario][view_name] = vqa_fp
    
    # Process external BDD_PC_5K scenarios
    for vqa_fp in external_vqa_paths:
        scenario = vqa_fp.stem.replace(".json", "")
        view_name = "single_view"
        
        if scenario not in external_scenarios:
            external_scenarios[scenario] = {}
        if view_name not in external_scenarios[scenario]:
            external_scenarios[scenario][view_name] = vqa_fp
    
    print(f"FILES: Found {len(scenarios)} main + {len(external_scenarios)} external scenarios for {split}")

    # Prepare tasks
    main_tasks = [
        (scenario, view_files, data_root, out_root, split, bbox_data, i, "main")
        for i, (scenario, view_files) in enumerate(scenarios.items())
    ]
    
    external_tasks = [
        (scenario, view_files, external_bdd_dir, out_root, split, external_bbox_data, i + len(scenarios), "external")
        for i, (scenario, view_files) in enumerate(external_scenarios.items())
    ]
    
    all_tasks = main_tasks + external_tasks
    
    print(f"STARTING: Processing {len(all_tasks)} VQA best view tasks...")
    all_samples = []
    
    with Pool(num_workers) as pool:
        results = pool.imap(process_scenario_task, all_tasks)
        
        for result in tqdm(results, total=len(all_tasks), ncols=100, 
                          desc=f"Processing VQA Best View {split}"):
            if result is not None:
                all_samples.extend(result)
    
    # Write results
    print(f"WRITING: Writing {len(all_samples)} VQA best view samples...")
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_jsonl, "w") as fout:
        for sample_data in all_samples:
            fout.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
    
    print(f"SUCCESS: {split} complete: {len(all_samples)} VQA best view samples written to {out_jsonl}")
    return len(all_samples)

def process_scenario_task(task_data):
    """Process a single scenario with best view selection for VQA training"""
    scenario, view_files, data_root, out_root, split, bbox_data, task_id, source = task_data
    
    try:
        print(f"PROCESSING: Processing VQA best view {source} scenario {scenario} with views: {list(view_files.keys())}")
        
        # Load VQA data for all views
        view_vqa_data = {}
        for view_name, vqa_fp in view_files.items():
            with open(vqa_fp) as f:
                vqa_json = json.load(f)
            
            # Handle different VQA JSON structures
            if isinstance(vqa_json, list):
                # External BDD_PC_5K format: list with single object
                if len(vqa_json) > 0:
                    vqa_data = vqa_json[0]
                else:
                    continue
            else:
                # Main WTS format: direct object
                vqa_data = vqa_json
            
            # Validate phases
            phases = vqa_data.get("event_phase", [])
            if len(phases) != 5:
                print(f"ERROR: Expected 5 phases, got {len(phases)} for {scenario}/{view_name}")
                continue
                
            # Build phase mapping
            phases_by_label = {}
            for i, phase in enumerate(phases):
                label_list = phase.get("labels", [])
                if label_list and isinstance(label_list[0], str):
                    phase_name = label_list[0]
                    if phase_name in PHASE_DESCRIPTIONS:
                        # Use the index from the VQA file as the phase number
                        phases_by_label[i] = phase
            
            if len(phases_by_label) != 5:
                print(f"    WARNING:  Phase mapping issue for {scenario}/{view_name}")
                print(f"      Expected 5 phases, got {len(phases_by_label)}: {list(phases_by_label.keys())}")
                # Continue processing with available phases
            
            view_vqa_data[view_name] = phases_by_label
        
        if not view_vqa_data:
            print(f"ERROR: No valid VQA data for scenario {scenario}")
            return None
        
        # Setup paths based on source
        if source == "external":
            videos_dir = Path(data_root) / "videos" / split
        elif source == "main":
            data_root = Path(data_root)
            if scenario.startswith("2023") and "normal" in scenario:
                videos_dir = data_root / "videos" / split / "normal_trimmed"
            else:
                videos_dir = data_root / "videos" / split
        else:
            videos_dir = Path(data_root) / "videos" / split
        
        all_samples = []
        
        # Process each phase (0-4) with best view selection
        for phase_num in range(5):
            if phase_num not in view_vqa_data[list(view_vqa_data.keys())[0]]:
                continue
                
            # Get the actual phase name from the VQA data
            first_view_data = view_vqa_data[list(view_vqa_data.keys())[0]]
            actual_phase_name = first_view_data[phase_num].get("labels", ["unknown"])[0]
            
            print(f"  TARGET: Processing Best View Phase {phase_num}: {actual_phase_name}")
            
            # Select the best view for this phase
            best_view = select_best_view_for_phase(phase_num, view_files, videos_dir, bbox_data, source, scenario)
            
            if not best_view:
                print(f"    ERROR: No suitable view found for phase {phase_num}")
                continue
            
            # Get conversations from the best view's VQA data
            best_view_vqa = view_vqa_data[best_view['view_name']]
            phase_conversations = best_view_vqa[phase_num].get("conversations", [])
            
            if not phase_conversations:
                print(f"    ERROR: No conversations found for phase {phase_num}")
                continue
            
            # Extract frame from best view
            if source == "external":
                rel_path = f"{split}/{scenario}/{best_view['camera_stem']}_phase{phase_num}_f{best_view['frame_choice']['frame']}.jpg"
            else:
                rel_path = f"{split}/{scenario}/{best_view['view_name']}_{best_view['camera_stem']}_phase{phase_num}_f{best_view['frame_choice']['frame']}.jpg"
            
            full_output_path = Path(out_root) / rel_path
            
            W, H, saved_paths = extract_and_save_frame_best_view(
                best_view['video_path'],
                best_view['frame_choice']['frame'],
                full_output_path,
                best_view['frame_choice']['ped_box'], 
                best_view['frame_choice']['veh_box'],
                best_view['view_name'],
                source,
                should_crop=True,
                save_full_too=True
            )
            
            if W is None or not saved_paths:
                print(f"    ERROR: Failed to extract frame for best view")
                continue
            
            # Create enhanced system prompt for VQA task
            system_prompt = create_vqa_system_prompt(actual_phase_name, PHASE_DESCRIPTIONS[actual_phase_name])
            
            # Create INDIVIDUAL samples for each Q&A pair using the best view images
            for i, conv in enumerate(phase_conversations):
                formatted_question, correct_answer = format_vqa_question(conv, "best_view")
                
                # Skip if we couldn't get a proper answer
                if not correct_answer:
                    print(f"      WARNING:  Skipping question {i+1} - no valid answer")
                    continue
                
                # Create image blocks based on available images (crop + full or just full)
                image_block = "\n".join(["<image>"] * len(saved_paths))
                
                # Create individual conversation with best view images
                conversations_list = [
                    {
                        "from": "human",
                        "value": f"{image_block}\n\n{formatted_question}"
                    },
                    {
                        "from": "gpt", 
                        "value": correct_answer
                    }
                ]
                
                # UPDATED JSON FORMAT: system field at same level as conversations
                sample_data = {
                    "conversations": conversations_list,
                    "system": system_prompt,
                    "images": saved_paths
                }
                
                all_samples.append(sample_data)
                
                print(f"      SUCCESS: Created best view VQA sample {i+1}/{len(phase_conversations)} for Phase {phase_num} with {len(saved_paths)} images from {best_view['camera_stem']}")
        
        # PROCESS ENVIRONMENT QUESTIONS
        print(f"  ENVIRONMENT: Processing Environment Questions for {scenario}")
        
        # Load environment questions
        if source == "external":
            scenario_dir = Path(data_root) / "annotations" / "vqa" / split / scenario
        else:
            if scenario.startswith("2023") and "normal" in scenario:
                scenario_dir = Path(data_root) / "annotations" / "vqa" / split / "normal_trimmed" / scenario
            else:
                scenario_dir = Path(data_root) / "annotations" / "vqa" / split / scenario
        
        environment_questions = load_environment_questions(scenario_dir, scenario, source)
        
        if environment_questions:
            # Filter out non-camera views (environment directory is not a camera view)
            camera_view_files = {k: v for k, v in view_files.items() if k not in ['environment']}
            
            # Select the best view for environment questions (prioritize vehicle view + largest image)
            env_best_view = select_best_view_for_environment(camera_view_files, videos_dir, bbox_data, source, scenario)
            
            if env_best_view:
                # Find the largest pedestrian bbox across all views for env_pedestrian
                largest_pedestrian_view = find_largest_pedestrian_bbox_across_views(camera_view_files, videos_dir, bbox_data, source, scenario)
                
                # Create env_largest image (vehicle view with largest bbox - full image)
                if source == "external":
                    env_largest_rel_path = f"{split}/{scenario}/env_largest_{env_best_view['camera_stem']}_f{env_best_view['frame_choice']['frame']}.jpg"
                else:
                    env_largest_rel_path = f"{split}/{scenario}/env_largest_{env_best_view['view_name']}_{env_best_view['camera_stem']}_f{env_best_view['frame_choice']['frame']}.jpg"
                
                env_largest_output_path = Path(out_root) / env_largest_rel_path
                
                W1, H1, env_largest_paths = extract_and_save_frame_best_view(
                    env_best_view['video_path'],
                    env_best_view['frame_choice']['frame'],
                    env_largest_output_path,
                    env_best_view['frame_choice']['ped_box'], 
                    env_best_view['frame_choice']['veh_box'],
                    env_best_view['view_name'],
                    source,
                    should_crop=False, # Full image for env_largest
                    save_full_too=True
                )
                
                # Create env_pedestrian image (best view with largest bbox - crop image only)
                env_pedestrian_paths = []
                if largest_pedestrian_view and largest_pedestrian_view != env_best_view:
                    if source == "external":
                        env_pedestrian_rel_path = f"{split}/{scenario}/env_pedestrian_{largest_pedestrian_view['camera_stem']}_f{largest_pedestrian_view['frame_choice']['frame']}.jpg"
                    else:
                        env_pedestrian_rel_path = f"{split}/{scenario}/env_pedestrian_{largest_pedestrian_view['view_name']}_{largest_pedestrian_view['camera_stem']}_f{largest_pedestrian_view['frame_choice']['frame']}.jpg"
                    
                    env_pedestrian_output_path = Path(out_root) / env_pedestrian_rel_path
                    
                    # Check if the largest pedestrian image has already been saved as part of env_largest_paths (full frame)
                    # We only need to save the crop here.
                    if env_largest_output_path.resolve() == env_pedestrian_output_path.resolve():
                         W2, H2, env_pedestrian_paths = extract_and_save_frame_best_view(
                            env_best_view['video_path'],
                            env_best_view['frame_choice']['frame'],
                            env_pedestrian_output_path,
                            env_best_view['frame_choice']['ped_box'], 
                            env_best_view['frame_choice']['veh_box'],
                            env_best_view['view_name'],
                            source,
                            should_crop=True,  # Crop image for env_pedestrian
                            save_full_too=False # Only save crop
                        )
                    else:
                        W2, H2, env_pedestrian_paths = extract_and_save_frame_best_view(
                            env_best_view['video_path'],
                            env_best_view['frame_choice']['frame'],
                            env_pedestrian_output_path,
                            env_best_view['frame_choice']['ped_box'], 
                            env_best_view['frame_choice']['veh_box'],
                            env_best_view['view_name'],
                            source,
                            should_crop=True,  # Crop image for env_pedestrian
                            save_full_too=False
                        )
                elif largest_pedestrian_view and largest_pedestrian_view == env_best_view:
                    # If the largest pedestrian view is the same as env_best_view, create crop from the same frame
                    if source == "external":
                        env_pedestrian_rel_path = f"{split}/{scenario}/env_pedestrian_{env_best_view['camera_stem']}_f{env_best_view['frame_choice']['frame']}.jpg"
                    else:
                        env_pedestrian_rel_path = f"{split}/{scenario}/env_pedestrian_{env_best_view['view_name']}_{env_best_view['camera_stem']}_f{env_best_view['frame_choice']['frame']}.jpg"
                    
                    env_pedestrian_output_path = Path(out_root) / env_pedestrian_rel_path
                    
                    # Check if the largest pedestrian image has already been saved as part of env_largest_paths (full frame)
                    # We only need to save the crop here.
                    if env_largest_output_path.resolve() == env_pedestrian_output_path.resolve():
                         W2, H2, env_pedestrian_paths = extract_and_save_frame_best_view(
                            env_best_view['video_path'],
                            env_best_view['frame_choice']['frame'],
                            env_pedestrian_output_path,
                            env_best_view['frame_choice']['ped_box'], 
                            env_best_view['frame_choice']['veh_box'],
                            env_best_view['view_name'],
                            source,
                            should_crop=True,  # Crop image for env_pedestrian
                            save_full_too=False # Only save crop
                        )
                    else:
                        W2, H2, env_pedestrian_paths = extract_and_save_frame_best_view(
                            env_best_view['video_path'],
                            env_best_view['frame_choice']['frame'],
                            env_pedestrian_output_path,
                            env_best_view['frame_choice']['ped_box'], 
                            env_best_view['frame_choice']['veh_box'],
                            env_best_view['view_name'],
                            source,
                            should_crop=True,  # Crop image for env_pedestrian
                            save_full_too=False
                        )
                
                # Combine all environment images (env_largest + env_pedestrian crop only)
                all_env_images = env_largest_paths + env_pedestrian_paths
                all_env_images = sorted(list(set(all_env_images))) # Remove duplicates
                
                if W1 is not None and env_largest_paths:
                    # Create environment system prompt
                    env_system_prompt = create_environment_system_prompt()
                    
                    # Create INDIVIDUAL samples for each environment Q&A pair
                    for i, env_question in enumerate(environment_questions):
                        formatted_question, correct_answer = format_vqa_question(env_question, "environment")
                        
                        # Skip if we couldn't get a proper answer
                        if not correct_answer:
                            print(f"      WARNING:  Skipping environment question {i+1} - no valid answer")
                            continue
                        
                        # Create image blocks based on available images (env_largest + env_pedestrian crop)
                        image_block = "\n".join(["<image>"] * len(all_env_images))
                        
                        # Create individual conversation for environment question
                        conversations_list = [
                            {
                                "from": "human",
                                "value": f"{image_block}\n\n{formatted_question}"
                            },
                            {
                                "from": "gpt", 
                                "value": correct_answer
                            }
                        ]
                        
                        # UPDATED JSON FORMAT: system field at same level as conversations
                        sample_data = {
                            "conversations": conversations_list,
                            "system": env_system_prompt,
                            "images": all_env_images
                        }
                        
                        all_samples.append(sample_data)
                        
                        print(f"      SUCCESS: Created environment VQA sample {i+1}/{len(environment_questions)} with {len(all_env_images)} images (env_largest: {env_best_view['view_name']}, env_pedestrian: {largest_pedestrian_view['view_name'] if largest_pedestrian_view else 'same'})")
                else:
                    print(f"    ERROR: Failed to extract environment frame from best view")
            else:
                print(f"    ERROR: No suitable view found for environment questions")
        else:
            print(f"    ℹ️  No environment questions found for scenario {scenario}")
        
        # Return all VQA samples for this scenario (phases + environment)
        return all_samples if all_samples else None
        
    except Exception as e:
        print(f"ERROR: Error processing scenario {scenario}: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_bbox_maps_fast(bbox_root, split):
    """Build bbox mappings for fast lookup (same as subtask 1)"""
    bbox_root = Path(bbox_root)
    if not bbox_root.exists():
        print(f"WARNING:  Bbox directory not found: {bbox_root}")
        return {}
    
    json_files = list(bbox_root.rglob("*.json"))
    if not json_files:
        return {}
    
    print(f"LOADING: Loading {len(json_files)} bbox files from {bbox_root}")
    
    mp = {}
    with Pool(min(16, len(json_files))) as pool:
        results = pool.map(process_bbox_file, json_files)
    
    # Combine results efficiently
    # New structure: mp[camera_stem] = {phase_num: (frame_id, bbox), ...}
    for stem, bbox_data in results:
        if bbox_data:
            if stem not in mp:
                mp[stem] = {}
            mp[stem].update(bbox_data)
    
    return mp

def process_bbox_file(js):
    """Helper for parallel bbox file processing"""
    stem = js.stem.replace("_bbox", "")
    return stem, coco2map_fast(js)

def find_largest_pedestrian_bbox_across_views(view_files, videos_dir, bbox_data, source, scenario):
    """Find the view with the largest pedestrian bbox across all views for env_pedestrian selection"""
    
    largest_view = None
    largest_area = 0
    
    # Collect all available views with their priorities
    for view_name, vqa_fp in view_files.items():
        try:
            if source == "external":
                # External data: single video file approach
                video_path = videos_dir / f"{scenario}.mp4"
                if not video_path.exists():
                    continue
                    
                camera_stem = scenario
                
                # Find the best frame with largest pedestrian bbox
                best_frame = find_best_environment_frame(video_path, camera_stem, bbox_data, source, avoid_frame_0=True)
                
                if best_frame and best_frame.get('ped_box'):
                    ped_box = best_frame['ped_box']
                    if ped_box and len(ped_box) >= 4:
                        bbox_area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                        
                        if bbox_area > largest_area:
                            largest_area = bbox_area
                            largest_view = {
                                'video_path': video_path,
                                'camera_stem': camera_stem,
                                'view_name': view_name,
                                'frame_choice': best_frame,
                                'pedestrian_bbox_area': bbox_area
                            }
            else:
                # Main WTS dataset with nested structure
                video_dir = videos_dir / scenario / view_name
                if not video_dir.exists():
                    continue
                    
                vids = sorted(video_dir.glob("*.mp4"))
                if not vids:
                    continue
                
                # Process each camera in this view
                for vid in vids:
                    camera_stem = vid.stem
                    
                    # Find the best frame with largest pedestrian bbox
                    best_frame = find_best_environment_frame(vid, camera_stem, bbox_data, source, avoid_frame_0=True)
                    
                    if best_frame and best_frame.get('ped_box'):
                        ped_box = best_frame['ped_box']
                        if ped_box and len(ped_box) >= 4:
                            bbox_area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                            
                            if bbox_area > largest_area:
                                largest_area = bbox_area
                                largest_view = {
                                    'video_path': vid,
                                    'camera_stem': camera_stem,
                                    'view_name': view_name,
                                    'frame_choice': best_frame,
                                    'pedestrian_bbox_area': bbox_area
                                }
        except Exception as e:
            print(f"    WARNING:  Error processing {view_name} for largest pedestrian search: {e}")
            continue
    
    if largest_view:
        print(f"    PEDESTRIAN: Found largest pedestrian bbox: {largest_area} pixels in {largest_view['view_name']}")
    
    return largest_view

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True, help="Root directory containing the WTS dataset")
    pa.add_argument("--out_root", required=True, help="Output directory for processed images")
    pa.add_argument("--out_jsonl", required=True, help="Output JSONL file path")
    pa.add_argument("--split", choices=["train", "val", "both"], default="both", help="Dataset split to process (default: both)")
    pa.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers (default: 32)")
    pa.add_argument("--merge_splits", action="store_true", help="Merge train and val splits into a single training dataset")
    args = pa.parse_args()

    print("STARTING: WTS Dataset VQA Best View Processor - SUBTASK 2 BEST VIEW + ENVIRONMENT")
    print("="*70)
    print("INFO: Configuration:")
    print(f"  • Data root: {args.data_root}")
    print(f"  • Output root: {args.out_root}")
    print(f"  • Output JSONL: {args.out_jsonl}")
    print(f"  • Split: {args.split}")
    print(f"  • Parallel workers: {args.num_workers}")
    print(f"  • Merge splits: {args.merge_splits}")
    print("STATS: Data Sources:")
    print(f"  • Main WTS VQA: annotations/vqa/{args.split if args.split != 'both' else '{train,val}'}/")
    print(f"  • Normal trimmed VQA: annotations/vqa/{args.split if args.split != 'both' else '{train,val}'}/normal_trimmed/")
    print(f"  • External BDD_PC_5K VQA: external/BDD_PC_5K/annotations/vqa/")
    print(f"  • Environment questions: annotations/vqa/{args.split if args.split != 'both' else '{train,val}'}/*/environment/")
    print("TARGET: Features:")
    print(f"  • Best view selection based on bbox availability and view priority")
    print(f"  • Phase-based questions: Select best view per phase (0-4)")
    print(f"  • Environment questions: Prioritize vehicle view + largest images, avoid frame 0")
    print(f"  • Max 2 images per sample: [crop, full] when bbox available or [full] when no bbox")
    print(f"  • No image resizing - original resolution preserved")
    print(f"  • Updated JSON format with system field at conversations level")
    print(f"  • Intelligent view prioritization for optimal VLM guidance")
    print("="*70)

    total_samples = 0
    overall_start = time.time()
    
    if args.split == "both":
        if args.merge_splits:
            # Merge train and val into a single dataset
            print("MERGE: MERGING train and val splits into single VQA best view training dataset")
            
            all_merged_samples = []
            
            for split in ["train", "val"]:
                print(f"\nFOLDER: Processing {split} split for VQA best view merging...")
                
                # Create temporary output for this split
                temp_jsonl = args.out_jsonl.replace(".jsonl", f"_temp_{split}.jsonl")
                split_samples = process_split_fast(args.data_root, args.out_root, temp_jsonl, split, args.num_workers)
                
                # Load the temporary file and add to merged samples
                if Path(temp_jsonl).exists():
                    with open(temp_jsonl, 'r') as f:
                        for line in f:
                            sample_data = json.loads(line.strip())
                            all_merged_samples.append(sample_data)
                    
                    # Remove temporary file
                    Path(temp_jsonl).unlink()
                
                print(f"SUCCESS: {split} complete: {split_samples} samples")
            
            # Write merged results
            print(f"WRITING: Writing {len(all_merged_samples)} merged VQA best view samples...")
            Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)  
            
            with open(args.out_jsonl, "w") as fout:
                for sample_data in all_merged_samples:
                    fout.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
            
            total_samples = len(all_merged_samples)
            
        else:
            # Original behavior: separate train and val files
            for split in ["train", "val"]:
                split_jsonl = args.out_jsonl.replace(".jsonl", f"_{split}.jsonl")
                samples = process_split_fast(args.data_root, args.out_root, split_jsonl, split, args.num_workers)
                total_samples += samples
    else:
        total_samples = process_split_fast(args.data_root, args.out_root, args.out_jsonl, args.split, args.num_workers)
    
    total_time = time.time() - overall_start
    
    print(f"\nCOMPLETE: VQA BEST VIEW + ENVIRONMENT PROCESSING COMPLETE!")
    print(f"STATS: FINAL PERFORMANCE:")
    print(f"  • Total VQA samples (phases + environment): {total_samples}")
    print(f"  • Total time: {total_time:.2f}s")
    print(f"  • Overall speed: {total_samples/total_time:.2f} samples/sec")
    if args.merge_splits:
        print(f"MERGE: MERGED VQA BEST VIEW + ENVIRONMENT TRAINING DATASET CREATED: {args.out_jsonl}")

if __name__ == "__main__":
    main() 