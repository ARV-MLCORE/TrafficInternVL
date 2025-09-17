#!/usr/bin/env python3
import json
import sys
import argparse

def count_json_entries(file_path):
    """Count entries in a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's a dictionary with video IDs as keys
        if isinstance(data, dict):
            # Check if it has metadata structure
            if 'metadata' in data and 'results' in data:
                # Old format with metadata
                results = data['results']
                if isinstance(results, list):
                    total_entries = len(results)
                    video_count = len(set(item.get('video_id', '') for item in results))
                    structure = "metadata_with_results"
                else:
                    total_entries = 0
                    video_count = 0
                    structure = "unknown"
            else:
                # New format: video_id as keys
                video_count = len(data)
                total_entries = sum(len(frames) for frames in data.values())
                structure = "video_id_to_frames"
        elif isinstance(data, list):
            # List format
            total_entries = len(data)
            video_count = len(set(item.get('video_id', '') for item in data))
            structure = "list_format"
        else:
            total_entries = 0
            video_count = 0
            structure = "unknown"
        
        return {
            'video_count': video_count,
            'total_entries': total_entries,
            'structure': structure
        }
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Count entries in a JSON file')
    parser.add_argument('file_path', help='Path to the JSON file')
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.file_path}")
    
    result = count_json_entries(args.file_path)
    
    if result:
        print(f"Structure: {result['structure']}")
        print(f"Number of videos: {result['video_count']}")
        print(f"Total entries: {result['total_entries']}")
    else:
        print("Failed to analyze the file")
        sys.exit(1)

if __name__ == "__main__":
    main()