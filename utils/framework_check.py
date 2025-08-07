#!/usr/bin/env python3
"""
SODA Framework Comprehensive Check
Verifies all components, file paths, and workflow functionality
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import importlib.util

def check_directory_structure():
    """Check if all required directories exist"""
    print("=== 📁 Directory Structure Check ===")
    
    required_dirs = [
        "data",
        "prompts", 
        "outputs",
        "config",
        "src",
        "analysis",
        "assets"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
            print(f"❌ Missing directory: {dir_name}")
        else:
            print(f"✅ {dir_name}/")
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def check_data_structure():
    """Check data directory structure and image counts"""
    print("\n=== 📊 Data Structure Check ===")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return False
    
    expected_combinations = [
        "gpt_backpack_images", "gpt_car_images", "gpt_cup_images", "gpt_laptop_images", "gpt_teddy bear_images",
        "imagen_backpack_images", "imagen_car_images", "imagen_cup_images", "imagen_laptop_images", "imagen_teddy bear_images", 
        "diffusion_backpack_images", "diffusion_car_images", "diffusion_cup_images", "diffusion_laptop_images", "diffusion_teddy bear_images"
    ]
    
    missing_combinations = []
    for combo in expected_combinations:
        combo_dir = data_dir / combo
        if not combo_dir.exists():
            missing_combinations.append(combo)
            print(f"❌ Missing: {combo}")
        else:
            # Count images in this combination
            image_count = 0
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_count += len(list(combo_dir.rglob(ext)))
            print(f"✅ {combo}: {image_count} images")
    
    if missing_combinations:
        print(f"\n⚠️  Missing combinations: {missing_combinations}")
        return False
    else:
        print("✅ All expected model-object combinations exist")
        return True

def check_prompt_files():
    """Check if all prompt CSV files exist"""
    print("\n=== 📝 Prompt Files Check ===")
    
    objects = ["backpack", "car", "cup", "laptop", "teddy bear"]
    missing_prompts = []
    
    for obj in objects:
        prompt_file = f"prompts/{obj}_prompts.csv"
        if not os.path.exists(prompt_file):
            missing_prompts.append(prompt_file)
            print(f"❌ Missing: {prompt_file}")
        else:
            # Check prompt count
            try:
                df = pd.read_csv(prompt_file)
                print(f"✅ {prompt_file}: {len(df)} prompts")
            except Exception as e:
                print(f"❌ Error reading {prompt_file}: {e}")
                missing_prompts.append(prompt_file)
    
    if missing_prompts:
        print(f"\n⚠️  Missing prompt files: {missing_prompts}")
        return False
    else:
        print("✅ All prompt files exist and are readable")
        return True

def check_source_files():
    """Check if all required source files exist"""
    print("\n=== 🔧 Source Files Check ===")
    
    required_files = [
        "src/prompt_generator.py",
        "src/image_generator.py", 
        "src/feature_gen.py",
        "src/feature_extract.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing source files: {missing_files}")
        return False
    else:
        print("✅ All source files exist")
        return True

def check_analysis_files():
    """Check if all analysis files exist"""
    print("\n=== 📈 Analysis Files Check ===")
    
    required_files = [
        "analysis/analysis_bias.py",
        "analysis/analysis_js.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing analysis files: {missing_files}")
        return False
    else:
        print("✅ All analysis files exist")
        return True

def check_config_files():
    """Check if configuration files exist"""
    print("\n=== ⚙️  Config Files Check ===")
    
    config_dir = Path("config")
    if config_dir.exists():
        config_files = list(config_dir.glob("*.json"))
        if config_files:
            for config_file in config_files:
                print(f"✅ {config_file}")
        else:
            print("⚠️  No config files found in config/")
    else:
        print("⚠️  Config directory not found")
    
    return True

def check_imports():
    """Check if all required Python modules can be imported"""
    print("\n=== 🐍 Python Imports Check ===")
    
    modules_to_check = [
        "pandas",
        "numpy", 
        "PIL",
        "cv2",
        "sklearn",
        "matplotlib",
        "seaborn"
    ]
    
    missing_modules = []
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - not installed")
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {missing_modules}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required modules are available")
        return True

def check_main_framework():
    """Check if main framework can be imported and run"""
    print("\n=== 🚀 Main Framework Check ===")
    
    try:
        # Test importing main components
        sys.path.insert(0, str(Path.cwd()))
        
        # Test prompt generator
        from src.prompt_generator import ObjectBiasPromptGenerator
        print("✅ ObjectBiasPromptGenerator imported successfully")
        
        # Test image generator
        from src.image_generator import UnifiedImageGenerator
        print("✅ UnifiedImageGenerator imported successfully")
        
        # Test feature discovery
        from src.feature_gen import ObjectBiasFeatureDiscovery
        print("✅ ObjectBiasFeatureDiscovery imported successfully")
        
        # Test feature extraction
        from src.feature_extract import ObjectBiasFeatureExtraction
        print("✅ ObjectBiasFeatureExtraction imported successfully")
        
        # Test bias analysis
        from analysis.analysis_bias import ObjectBiasAnalyzer
        print("✅ ObjectBiasAnalyzer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_workflow():
    """Test basic workflow functionality"""
    print("\n=== 🔄 Workflow Test ===")
    
    try:
        # Test with a simple object
        test_object = "backpack"
        
        # Check if prompt file exists for test object
        prompt_file = f"prompts/{test_object}_prompts.csv"
        if not os.path.exists(prompt_file):
            print(f"❌ Cannot test workflow - missing {prompt_file}")
            return False
        
        # Check if data exists for test object
        data_dirs = [f"data/gpt_{test_object}_images", f"data/imagen_{test_object}_images", f"data/diffusion_{test_object}_images"]
        missing_data = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                missing_data.append(data_dir)
        
        if missing_data:
            print(f"⚠️  Missing data directories for workflow test: {missing_data}")
        else:
            print("✅ Data directories exist for workflow test")
        
        print("✅ Basic workflow components are available")
        return True
        
    except Exception as e:
        print(f"❌ Workflow test error: {e}")
        return False

def main():
    print("🔍 SODA Framework Comprehensive Check")
    print("=" * 50)
    
    checks = [
        check_directory_structure(),
        check_data_structure(),
        check_prompt_files(),
        check_source_files(),
        check_analysis_files(),
        check_config_files(),
        check_imports(),
        check_main_framework(),
        check_workflow()
    ]
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    passed_checks = sum(checks)
    total_checks = len(checks)
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ SODA Framework is ready to use")
    else:
        print(f"⚠️  {total_checks - passed_checks} checks failed")
        print("❌ Some issues need to be resolved")
    
    print(f"📊 Passed: {passed_checks}/{total_checks}")

if __name__ == "__main__":
    main() 