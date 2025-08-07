#!/usr/bin/env python3
"""
SODA Baseline vs Demographics Score Analysis
Calculates JS divergence between baseline and demographic groups
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import os

class BaselineVsDemographicsAnalyzer:
    def __init__(self, base_output_dir="/home/allsound/SODA/outputs"):
        # Check if model_results subdirectory exists and use it
        base_path = Path(base_output_dir)
        if base_path.exists():
            images_path = base_path / "images"
            if images_path.exists():
                self.base_output_dir = images_path
            else:
                self.base_output_dir = base_path
        self.results = {}
        
        # Auto-detect available models and objects
        self.models, self.objects = self.detect_available_models_and_objects()
        
        # 8 demographic groups
        self.demographic_groups = [
            "young_adults", "middle_aged", "elderly",  # Age (3)
            "men", "women",                            # Gender (2)
            "white", "black", "asian"                  # Ethnicity (3)
        ]
        
        
        print(f"Analysis target folder: {self.base_output_dir}")
        print(f"Detected models: {', '.join(self.models)}")
        print(f"Detected objects: {', '.join(self.objects)}")
        print(f"8 Demographics: {', '.join(self.demographic_groups)}")
        print(f"Including all features - no exclusions")
    
    def detect_available_models_and_objects(self):
        """Automatically detect available models and objects from the output directory"""
        models = set()
        objects = set()
        
        if not self.base_output_dir.exists():
            print(f"Warning: Base output directory does not exist: {self.base_output_dir}")
            return [], []
        
        # Scan for folders with pattern: {model}_{object}_images
        for folder in self.base_output_dir.iterdir():
            if folder.is_dir() and folder.name.endswith('_images'):
                # Extract model and object from folder name
                parts = folder.name.replace('_images', '').split('_')
                if len(parts) >= 2:
                    # Handle cases like "sdxl_car_images" -> model="sdxl", object="car"
                    # or "imagen4_car_images" -> model="imagen4", object="car"
                    if len(parts) == 2:
                        model, obj = parts
                    else:
                        # Handle multi-part model names like "imagen4"
                        model = '_'.join(parts[:-1])
                        obj = parts[-1]
                    
                    # Check if results folder exists
                    results_folder = folder / "results"
                    if results_folder.exists():
                        models.add(model)
                        objects.add(obj)
        
        return sorted(list(models)), sorted(list(objects))
        
    def load_analysis_results(self):
        """Load analysis results for all model-object combinations"""
        print("\n" + "="*60)
        print("Loading analysis results...")
        print("="*60)
        
        loaded_count = 0
        total_count = len(self.models) * len(self.objects)
        
        for model in self.models:
            for object_name in self.objects:
                folder_name = f"{model}_{object_name}_images"
                folder_path = self.base_output_dir / folder_name
                
                if not folder_path.exists():
                    print(f"Folder not found: {folder_name}")
                    continue
                
                # Load JS divergence analysis results
                js_file = folder_path / "results" / "js_divergence_analysis.json"
                
                if js_file.exists():
                    try:
                        with open(js_file, 'r', encoding='utf-8') as f:
                            js_data = json.load(f)
                        
                        self.results[folder_name] = {
                            'js_analysis': js_data,
                            'model': model,
                            'object': object_name
                        }
                        
                        loaded_count += 1
                        print(f"Loaded: {folder_name}")
                        
                    except Exception as e:
                        print(f"Failed to load: {folder_name} - {e}")
                else:
                    print(f"JS divergence file not found: {folder_name}")
        
        print(f"\nLoaded: {loaded_count}/{total_count} combinations")
        return loaded_count > 0
    
    def should_include_feature(self, feature_name):
        """Include all features - no exclusions"""
        return True
    
    def extract_baseline_vs_demographic_divergence(self, js_analysis, debug=False):
        """Extract JS divergence between baseline and each demographic group"""
        baseline_vs_group_results = js_analysis.get('baseline_vs_demographic_divergence', {})
        
        demographic_divergences = {}
        feature_info = defaultdict(int)  # For debugging
        
        for demo_type, groups in baseline_vs_group_results.items():
            for group_name, features in groups.items():
                # Include all features
                included_divergences = []
                
                for feature, result in features.items():
                    js_divergence = result.get('js_divergence', 0)
                    included_divergences.append(js_divergence)
                    feature_info[feature] += 1
                
                if included_divergences:
                    # Average JS divergence for this demographic group
                    avg_divergence = np.mean(included_divergences)
                    demographic_divergences[group_name] = avg_divergence
        
        # Debug information output (only for first analysis)
        if debug:
            print(f"   Included features: {len(set(feature_info.keys()))}")
            feature_types = {
                'product': [f for f in feature_info.keys() if f.startswith('product_')],
                'background': [f for f in feature_info.keys() if f.startswith('background_')],
                'color': [f for f in feature_info.keys() if 'color' in f.lower()]
            }
            for ftype, flist in feature_types.items():
                if flist:
                    print(f"   - {ftype}: {len(flist)} features ({', '.join(flist[:3])}...)")
        
        return demographic_divergences
    
    def create_baseline_vs_demographics_table(self):
        """Create main table: Baseline vs 8 Demographics JS Divergence"""
        print("\n" + "="*60)
        print("Creating main table: Baseline vs 8 Demographics JS Divergence")
        print("Including all features - no exclusions")
        print("="*60)
        
        table_data = []
        first_analysis = True
        
        for folder_name, data in self.results.items():
            model = data['model']
            object_name = data['object']
            js_analysis = data['js_analysis']
            
            # Debug information only for first analysis
            debug = first_analysis
            first_analysis = False
            
            # Extract baseline vs demographic divergence
            demographic_divergences = self.extract_baseline_vs_demographic_divergence(js_analysis, debug)
            
            # Create table row data
            row_data = {
                'Model': model.upper(),
                'Object': object_name.title()
            }
            
            # Add divergence for each of 8 demographics
            for demo_group in self.demographic_groups:
                divergence = demographic_divergences.get(demo_group, 0.0)
                row_data[demo_group.title().replace('_', ' ')] = round(divergence, 3)
            
            table_data.append(row_data)
            
            # Simple output
            print(f"{model.upper()} {object_name.title()}: "
                  f"avg divergence = {np.mean(list(demographic_divergences.values())):.3f}")
        
        df = pd.DataFrame(table_data)
        print(f"\nTable created: {len(df)} rows × {len(df.columns)} columns")
        
        return df
    
    def save_results(self, baseline_vs_demographics_df):
        """Save results"""
        print("\n" + "="*60)
        print("Saving results")
        print("="*60)
        
        try:
            # Use comprehensive_analysis directory - always use outputs/comprehensive_analysis
            output_dir = Path("outputs/comprehensive_analysis/bds_score")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main table
            output_file = output_dir / "baseline_vs_8_demographics_table.csv"
            baseline_vs_demographics_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
                
        except Exception as e:
            print(f"Error during save: {e}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self):
        """Run complete analysis"""
        print("Baseline vs Demographics Analysis")
        print("="*60)
        
        # 1. Load data
        if not self.load_analysis_results():
            print("Data loading failed")
            return False
        
        # 2. Create table
        baseline_df = self.create_baseline_vs_demographics_table()
        
        # 3. Save results
        self.save_results(baseline_df)
        
        print("\nAnalysis completed!")
        return True

# Execute
if __name__ == "__main__":
    analyzer = BaselineVsDemographicsAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\nAll analysis completed!")
    else:
        print("\nAnalysis failed")