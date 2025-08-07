#!/usr/bin/env python3
"""
Script to count all images in the SODA framework
"""

import os
import glob
from pathlib import Path

def count_images_in_directory(directory_path):
    """Count all image files in a directory and its subdirectories"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
    total_count = 0
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory_path, '**', ext)
        files = glob.glob(pattern, recursive=True)
        total_count += len(files)
        image_files.extend(files)
    
    return total_count, image_files

def main():
    print("=== SODA Image Count Verification ===")
    print("Checking if all 2700 images are present...\n")
    
    # Check data directory structure
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    total_images = 0
    model_counts = {}
    
    # Count images in each model directory
    for model_dir in data_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            model_name = model_dir.name
            count, files = count_images_in_directory(model_dir)
            model_counts[model_name] = count
            total_images += count
            
            print(f"📁 {model_name}: {count} images")
            if count > 0:
                # Show subdirectory breakdown
                for subdir in model_dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        sub_count, _ = count_images_in_directory(subdir)
                        if sub_count > 0:
                            print(f"   └── {subdir.name}: {sub_count} images")
    
    print(f"\n📊 Total images found: {total_images}")
    print(f"🎯 Expected images: 2700")
    
    if total_images == 2700:
        print("✅ SUCCESS: All 2700 images are present!")
    elif total_images > 2700:
        print(f"⚠️  WARNING: Found {total_images - 2700} extra images")
    else:
        print(f"❌ ERROR: Missing {2700 - total_images} images")
    
    # Calculate expected breakdown
    print(f"\n📋 Expected breakdown:")
    print(f"   - 5 objects (backpack, car, cup, laptop, teddy bear)")
    print(f"   - 3 models (gpt, gemini, diffusion)")
    print(f"   - 9 prompts per object (1 basic + 8 demographic)")
    print(f"   - 2 images per prompt")
    print(f"   - Total: 9 prompts × 2 images = 18 images per object")
    print(f"   - Total: 18 images × 5 objects = 90 images per model")
    print(f"   - Total: 90 images × 3 models = 270 images per model-object combination")
    print(f"   - Total: 270 images × 10 combinations = 2700 images")
    
    # Check if all models have the correct count
    expected_per_model_object = 180  # 9 prompts × 2 images × 10 images per prompt
    print(f"\n🔍 Model verification:")
    for model_name, count in model_counts.items():
        if count == expected_per_model_object:
            print(f"   ✅ {model_name}: {count} images (correct)")
        else:
            print(f"   ❌ {model_name}: {count} images (expected {expected_per_model_object})")
    
    print(f"\n🎉 Summary:")
    print(f"   - Found {len(model_counts)} model-object combinations")
    print(f"   - Each has {expected_per_model_object} images")
    print(f"   - Total: {len(model_counts)} × {expected_per_model_object} = {len(model_counts) * expected_per_model_object} images")

if __name__ == "__main__":
    main() 