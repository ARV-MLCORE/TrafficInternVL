#!/usr/bin/env python3
"""
Script to aggregate inference results from multiple JSON files into a single JSON file.
This script follows the structure expected by inference_image_input.py
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def aggregate_results(backup_dir: str, output_file: str) -> None:
    """
    Aggregate all JSON results from backup directory into a single JSON file.
    Format: {video_id: [frame1_data, frame2_data, ...]}
    
    Args:
        backup_dir: Directory containing video folders with JSON results
        output_file: Output path for the aggregated JSON file
    """
    
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        print(f"Error: Backup directory {backup_dir} does not exist!")
        return
    
    aggregated_results = {}
    total_files = 0
    processed_files = 0
    
    # Get all video directories
    video_dirs = [d for d in backup_path.iterdir() if d.is_dir()]
    video_dirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        print(f"Processing {video_name}...")
        
        # Get all JSON files in this video directory
        json_files = list(video_dir.glob("*.json"))
        json_files.sort(key=lambda x: int(x.stem))  # Sort by frame number
        
        total_files += len(json_files)
        
        # Initialize array for this video
        video_frames = []
        
        for json_file in json_files:
            frame_data = load_json_file(str(json_file))
            
            if frame_data:  # Only add if successfully loaded
                # Remove metadata fields if they exist (keep only labels, caption_vehicle, caption_pedestrian)
                clean_frame_data = {}
                if 'labels' in frame_data:
                    clean_frame_data['labels'] = frame_data['labels']
                if 'caption_vehicle' in frame_data:
                    clean_frame_data['caption_vehicle'] = frame_data['caption_vehicle']
                if 'caption_pedestrian' in frame_data:
                    clean_frame_data['caption_pedestrian'] = frame_data['caption_pedestrian']
                
                video_frames.append(clean_frame_data)
                processed_files += 1
        
        # Add video frames to results if any frames were processed
        if video_frames:
            aggregated_results[video_name] = video_frames
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Successfully aggregated results!")
        print(f"📊 Statistics:")
        print(f"   - Total videos: {len(video_dirs)}")
        print(f"   - Total frames processed: {processed_files}")
        print(f"   - Total files found: {total_files}")
        print(f"   - Output file: {output_file}")
        print(f"   - File size: {os.path.getsize(output_file):,} bytes")
        
    except Exception as e:
        print(f"❌ Error saving aggregated results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Aggregate inference results into a single JSON file')
    parser.add_argument('--backup_dir', required=True, 
                       help='Directory containing video folders with JSON results')
    parser.add_argument('--output_file', required=True,
                       help='Output path for the aggregated JSON file')
    
    args = parser.parse_args()
    
    print("🔄 Starting result aggregation...")
    print(f"📁 Backup directory: {args.backup_dir}")
    print(f"📄 Output file: {args.output_file}")
    print("-" * 50)
    
    aggregate_results(args.backup_dir, args.output_file)

if __name__ == "__main__":
    main()