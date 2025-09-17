#!/usr/bin/env python3
import json
import sys
import argparse

def load_json_keys(file_path):
    """Load JSON file and return set of keys"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Check if it has metadata structure
            if 'metadata' in data and 'results' in data:
                # Extract video_ids from results
                results = data['results']
                if isinstance(results, list):
                    keys = set(item.get('video_id', '') for item in results if item.get('video_id'))
                else:
                    keys = set()
            else:
                # Direct video_id keys
                keys = set(data.keys())
        else:
            keys = set()
        
        return keys
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def compare_keys(file1_path, file2_path):
    """Compare keys between two JSON files"""
    print(f"Loading keys from: {file1_path}")
    keys1 = load_json_keys(file1_path)
    print(f"Found {len(keys1)} keys in file 1")
    
    print(f"\nLoading keys from: {file2_path}")
    keys2 = load_json_keys(file2_path)
    print(f"Found {len(keys2)} keys in file 2")
    
    # Find common keys
    common_keys = keys1.intersection(keys2)
    print(f"\n🔍 Common keys: {len(common_keys)}")
    
    # Find unique keys
    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1
    
    print(f"📊 Keys only in file 1: {len(only_in_file1)}")
    print(f"📊 Keys only in file 2: {len(only_in_file2)}")
    
    # Show some examples
    if common_keys:
        print(f"\n✅ Sample common keys (first 10):")
        for i, key in enumerate(sorted(common_keys)[:10]):
            print(f"  {i+1}. {key}")
        if len(common_keys) > 10:
            print(f"  ... and {len(common_keys) - 10} more")
    
    if only_in_file1:
        print(f"\n🔴 Keys only in file 1 (first 10):")
        for i, key in enumerate(sorted(only_in_file1)[:10]):
            print(f"  {i+1}. {key}")
        if len(only_in_file1) > 10:
            print(f"  ... and {len(only_in_file1) - 10} more")
    
    if only_in_file2:
        print(f"\n🔵 Keys only in file 2 (first 10):")
        for i, key in enumerate(sorted(only_in_file2)[:10]):
            print(f"  {i+1}. {key}")
        if len(only_in_file2) > 10:
            print(f"  ... and {len(only_in_file2) - 10} more")
    
    # Calculate similarity
    total_unique_keys = len(keys1.union(keys2))
    similarity = len(common_keys) / total_unique_keys * 100 if total_unique_keys > 0 else 0
    print(f"\n📈 Key similarity: {similarity:.2f}%")
    
    return {
        'common_keys': len(common_keys),
        'only_in_file1': len(only_in_file1),
        'only_in_file2': len(only_in_file2),
        'similarity': similarity
    }

def main():
    parser = argparse.ArgumentParser(description='Compare keys between two JSON files')
    parser.add_argument('--file1', required=True, help='Path to first JSON file')
    parser.add_argument('--file2', required=True, help='Path to second JSON file')
    
    args = parser.parse_args()
    
    result = compare_keys(args.file1, args.file2)
    
    if result['common_keys'] == 0 and result['only_in_file1'] == 0 and result['only_in_file2'] == 0:
        print("❌ Failed to compare files")
        sys.exit(1)

if __name__ == "__main__":
    main()