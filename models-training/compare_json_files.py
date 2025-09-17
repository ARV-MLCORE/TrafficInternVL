#!/usr/bin/env python3
"""
Script to compare two JSON files and count the number of entries/records
"""

import json
import argparse
from pathlib import Path

def count_json_entries(file_path: str) -> dict:
    """
    Count entries in a JSON file and return statistics
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with statistics about the JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'file_path': file_path,
            'file_exists': True,
            'total_keys': 0,
            'total_entries': 0,
            'structure_type': 'unknown'
        }
        
        if isinstance(data, dict):
            stats['total_keys'] = len(data.keys())
            
            # Check if it's the new format (video_id: [frames])
            if all(isinstance(v, list) for v in data.values()):
                stats['structure_type'] = 'video_id_to_frames'
                stats['total_entries'] = sum(len(frames) for frames in data.values())
                stats['video_ids'] = list(data.keys())[:5]  # Show first 5 video IDs
                
            # Check if it's the old format with metadata and results
            elif 'results' in data and isinstance(data['results'], list):
                stats['structure_type'] = 'metadata_with_results'
                stats['total_entries'] = len(data['results'])
                if 'metadata' in data:
                    stats['metadata'] = data['metadata']
                    
            else:
                stats['structure_type'] = 'other_dict'
                stats['total_entries'] = len(data)
                
        elif isinstance(data, list):
            stats['structure_type'] = 'list'
            stats['total_entries'] = len(data)
            
        return stats
        
    except FileNotFoundError:
        return {
            'file_path': file_path,
            'file_exists': False,
            'error': 'File not found'
        }
    except json.JSONDecodeError as e:
        return {
            'file_path': file_path,
            'file_exists': True,
            'error': f'JSON decode error: {e}'
        }
    except Exception as e:
        return {
            'file_path': file_path,
            'file_exists': True,
            'error': f'Unexpected error: {e}'
        }

def compare_files(file1: str, file2: str):
    """Compare two JSON files and print statistics"""
    
    print("🔍 Comparing JSON files...")
    print("=" * 60)
    
    # Get statistics for both files
    stats1 = count_json_entries(file1)
    stats2 = count_json_entries(file2)
    
    # Print file 1 statistics
    print(f"\n📄 File 1: {Path(file1).name}")
    print(f"   Path: {file1}")
    if stats1['file_exists']:
        if 'error' not in stats1:
            print(f"   Structure: {stats1['structure_type']}")
            print(f"   Total keys: {stats1['total_keys']}")
            print(f"   Total entries: {stats1['total_entries']}")
            if 'video_ids' in stats1:
                print(f"   Sample video IDs: {stats1['video_ids']}")
            if 'metadata' in stats1:
                print(f"   Metadata: {stats1['metadata']}")
        else:
            print(f"   ❌ Error: {stats1['error']}")
    else:
        print(f"   ❌ File not found")
    
    # Print file 2 statistics
    print(f"\n📄 File 2: {Path(file2).name}")
    print(f"   Path: {file2}")
    if stats2['file_exists']:
        if 'error' not in stats2:
            print(f"   Structure: {stats2['structure_type']}")
            print(f"   Total keys: {stats2['total_keys']}")
            print(f"   Total entries: {stats2['total_entries']}")
            if 'video_ids' in stats2:
                print(f"   Sample video IDs: {stats2['video_ids']}")
            if 'metadata' in stats2:
                print(f"   Metadata: {stats2['metadata']}")
        else:
            print(f"   ❌ Error: {stats2['error']}")
    else:
        print(f"   ❌ File not found")
    
    # Compare results
    print(f"\n🔄 Comparison Results:")
    print("=" * 60)
    
    if (stats1['file_exists'] and stats2['file_exists'] and 
        'error' not in stats1 and 'error' not in stats2):
        
        entries1 = stats1['total_entries']
        entries2 = stats2['total_entries']
        
        if entries1 == entries2:
            print(f"✅ MATCH: Both files have {entries1} entries")
        else:
            print(f"❌ MISMATCH:")
            print(f"   File 1: {entries1} entries")
            print(f"   File 2: {entries2} entries")
            print(f"   Difference: {abs(entries1 - entries2)} entries")
            
        # Compare structure types
        if stats1['structure_type'] == stats2['structure_type']:
            print(f"✅ Structure types match: {stats1['structure_type']}")
        else:
            print(f"⚠️  Different structure types:")
            print(f"   File 1: {stats1['structure_type']}")
            print(f"   File 2: {stats2['structure_type']}")
            
    else:
        print("❌ Cannot compare due to file errors")

def main():
    parser = argparse.ArgumentParser(description='Compare two JSON files and count entries')
    parser.add_argument('--file1', required=True, help='Path to first JSON file')
    parser.add_argument('--file2', required=True, help='Path to second JSON file')
    
    args = parser.parse_args()
    
    compare_files(args.file1, args.file2)

if __name__ == "__main__":
    main()