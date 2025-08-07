#!/usr/bin/env python3
"""
SODA Visual Attribute Consistency Score Analysis
Model-centric bias analysis with perfect segregation and dramatic shifts
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import os

class ModelCentricDataExtractor:
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
        
        # Fixed features that should be included
        self.fixed_features = ['product_color', 'background_color']
        
        # Feature patterns to exclude (text-related only)
        self.exclude_patterns = [
            # 'text_presence',
            # 'background_text_presence'
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
        print("📊 Loading analysis results...")
        print("="*60)
        
        loaded_count = 0
        
        for model in self.models:
            for object_name in self.objects:
                folder_name = f"{model}_{object_name}_images"
                folder_path = self.base_output_dir / folder_name
                
                if not folder_path.exists():
                    print(f"⚠️ Folder not found: {folder_name}")
                    continue
                
                bias_file = folder_path / "results" / "comprehensive_bias_analysis.json"
                
                if bias_file.exists():
                    try:
                        with open(bias_file, 'r', encoding='utf-8') as f:
                            bias_data = json.load(f)
                        
                        self.results[folder_name] = {
                            'bias_analysis': bias_data,
                            'model': model,
                            'object': object_name
                        }
                        
                        loaded_count += 1
                        print(f"✅ {folder_name}")
                        
                    except Exception as e:
                        print(f"❌ {folder_name}: {e}")
        
        print(f"\n📊 Total {loaded_count} combinations loaded")
        return loaded_count > 0
    
    def calculate_overall_bias_score(self, bias_analysis, debug=False):
        """Calculate overall VAC score (including color, excluding text)"""
        all_bias_scores = []
        feature_scores = {}

        # Baseline bias - include all features
        baseline_stats = bias_analysis.get('baseline_bias_analysis', {})
        for feature, stats in baseline_stats.items():
            bias_score = stats.get('bias_score', 0)
            if bias_score >= 0.05:
                all_bias_scores.append(bias_score)
                feature_scores[f"baseline_{feature}"] = bias_score

        # Demographic bias - include product features or fixed features
        demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
        for demo_type, features_data in demographic_stats.items():
            for feature_name, groups_data in features_data.items():
                if (feature_name.startswith('product_') or feature_name in self.fixed_features):
                    for group_name, group_stats in groups_data.items():
                        bias_score = group_stats.get('bias_score', 0)
                        if bias_score >= 0.05:
                            all_bias_scores.append(bias_score)
                            feature_scores[f"{demo_type}_{group_name}_{feature_name}"] = bias_score

        if debug:
            print(f"  📊 Included features count: {len(feature_scores)}")
            print(f"  📊 Average VAC score: {np.mean(all_bias_scores) if all_bias_scores else 0:.3f}")
            color_scores = {k: v for k, v in feature_scores.items() if 'color' in k.lower()}
            if color_scores:
                print(f"  🎨 Color-related scores: {len(color_scores)}")
                for k, v in list(color_scores.items())[:5]:
                    print(f"     {k}: {v:.3f}")

        return np.mean(all_bias_scores) if all_bias_scores else 0.0

    
    def extract_model_bias_data(self):
        """Extract model-specific VAC score data (enhanced)"""
        print("\n" + "="*60)
        print("📊 Model VAC Score data extraction (including Color)")
        print("="*60)
        
        results = []
        
        for model in self.models:
            print(f"\n🤖 {model.upper()}:")
            model_scores = []
            
            for object_name in self.objects:
                folder_name = f"{model}_{object_name}_images"
                
                if folder_name in self.results:
                    bias_analysis = self.results[folder_name]['bias_analysis']
                    
                    # Debug mode - detailed output only for first object
                    debug = (object_name == self.objects[0])
                    bias_score = self.calculate_overall_bias_score(bias_analysis, debug=debug)
                    
                    model_scores.append(bias_score)
                    
                    print(f"   {object_name.title()}: {bias_score:.3f}")
                    
                    results.append({
                        'Model': model.upper(),
                        'Object': object_name.title(),
                        'VAC_Score': round(bias_score, 3)
                    })
                else:
                    print(f"   {object_name.title()}: No data")
                    results.append({
                        'Model': model.upper(),
                        'Object': object_name.title(), 
                        'VAC_Score': 0.0
                    })
            
            # Model average
            if model_scores:
                avg_score = np.mean(model_scores)
                print(f"   → Average: {avg_score:.3f}")
        
        return results
    
    def analyze_feature_distribution(self):
        """Analyze which features are included"""
        print("\n" + "="*60)
        print("📊 Feature Distribution Analysis")
        print("="*60)
        
        for model in self.models[:1]:  # Sample with first model only
            for object_name in self.objects[:1]:  # Sample with first object only
                folder_name = f"{model}_{object_name}"
                
                if folder_name in self.results:
                    bias_analysis = self.results[folder_name]['bias_analysis']
                    
                    print(f"\n🔍 {model.upper()} - {object_name.title()} Feature Analysis:")
                    
                    # Baseline features
                    baseline_features = list(bias_analysis.get('baseline_bias_analysis', {}).keys())
                    print(f"  Baseline features ({len(baseline_features)}):")
                    for feat in baseline_features:
                        print(f"    - {feat}")
                    
                    # Demographic features
                    demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
                    if demographic_stats:
                        first_demo = list(demographic_stats.keys())[0]
                        demo_features = list(demographic_stats[first_demo].keys())
                        print(f"  Demographic features ({len(demo_features)}):")
                        for feat in demo_features:
                            print(f"    - {feat}")
                    
                    break
    
    def create_pivot_table(self, results):
        """Create pivot table (model×object)"""
        df = pd.DataFrame(results)
        pivot_df = df.pivot(index='Model', columns='Object', values='VAC_Score')
        pivot_df['Average'] = pivot_df.mean(axis=1)
        return pivot_df
    
    def extract_vac_score_details(self, bias_analysis):
        """Extract all VAC score details"""
        all_scores = []
        score_details = []
        
        # 1. Baseline VAC scores
        baseline_stats = bias_analysis.get('baseline_bias_analysis', {})
        for feature, stats in baseline_stats.items():
            if not any(pattern in feature for pattern in self.exclude_patterns):
                vac_score = stats.get('bias_score', 0)
                distribution = stats.get('distribution', {})
                
                score_details.append({
                    'source': 'baseline',
                    'feature': feature,
                    'score': vac_score,
                    'context': 'baseline',
                    'distribution': distribution
                })
                all_scores.append(vac_score)
        
        # 2. Demographic VAC scores
        demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
        for demo_type, features_data in demographic_stats.items():
            for feature_name, groups_data in features_data.items():
                if (feature_name.startswith('product_') and 
                    not any(pattern in feature_name for pattern in self.exclude_patterns)):
                    
                    for group_name, group_stats in groups_data.items():
                        vac_score = group_stats.get('bias_score', 0)
                        distribution = group_stats.get('distribution', {})
                        
                        score_details.append({
                            'source': 'demographic',
                            'feature': feature_name,
                            'score': vac_score,
                            'context': f"{demo_type}_{group_name}",
                            'demo_type': demo_type,
                            'group': group_name,
                            'distribution': distribution
                        })
                        all_scores.append(vac_score)
        
        return all_scores, score_details
    
    def find_perfect_segregation_cases(self):
        """Find perfect segregation cases (score = 1.0)"""
        print("\n" + "="*60)
        print("🔍 Perfect Segregation (Score = 1.0) Case Analysis")
        print("="*60)
        
        perfect_cases = []
        model_perfect_counts = {model: 0 for model in self.models}
        
        for folder_name, data in self.results.items():
            model = data['model']
            object_name = data['object']
            bias_analysis = data['bias_analysis']
            
            # Extract VAC scores
            _, score_details = self.extract_vac_score_details(bias_analysis)
            
            # Find perfect segregation (strictly score == 1.0)
            perfect_scores = [item for item in score_details if item['score'] == 1.0]
            
            for item in perfect_scores:
                # Find the most dominant value from distribution
                distribution = item.get('distribution', {})
                if distribution:
                    dominant_value = max(distribution.items(), key=lambda x: x[1])
                    
                    perfect_case = {
                        'model': model.upper(),
                        'object': object_name.title(),
                        'feature': item['feature'].replace('product_', ''),
                        'context': item['context'],
                        'score': round(item['score'], 3),
                        'dominant_value': dominant_value[0],
                        'dominance_rate': round(dominant_value[1] / sum(distribution.values()), 3),
                        'distribution': distribution
                    }
                    perfect_cases.append(perfect_case)
                    model_perfect_counts[model] += 1
            
            print(f"🔍 {model.upper()} {object_name.title()}: {len(perfect_scores)} perfect cases")
        
        # Model-specific perfect segregation statistics
        print(f"\n📊 Model-specific Perfect Segregation Statistics:")
        total_perfect = sum(model_perfect_counts.values())
        for model, count in model_perfect_counts.items():
            print(f"   {model.upper()}: {count} times ({count/total_perfect*100:.1f}%)")
        
        return perfect_cases, model_perfect_counts
    
    def analyze_baseline_vs_demographic_shifts(self):
        """Analyze L1 vs L2 shift patterns"""
        print("\n" + "="*60)
        print("🔄 Baseline vs Demographic Shifts Analysis")
        print("="*60)
        
        shift_patterns = []
        
        for folder_name, data in self.results.items():
            model = data['model']
            object_name = data['object']
            bias_analysis = data['bias_analysis']
            
            # Baseline distributions
            baseline_stats = bias_analysis.get('baseline_bias_analysis', {})
            
            # Demographic distributions
            demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
            
            for feature_name in baseline_stats.keys():
                if (feature_name.startswith('product_') and 
                    not any(pattern in feature_name for pattern in self.exclude_patterns)):
                    
                    baseline_dist = baseline_stats[feature_name].get('distribution', {})
                    if not baseline_dist:
                        continue
                    
                    baseline_dominant = max(baseline_dist.items(), key=lambda x: x[1]) if baseline_dist else (None, 0)
                    
                    # Compare with each demographic group
                    for demo_type, features_data in demographic_stats.items():
                        if feature_name in features_data:
                            for group_name, group_stats in features_data[feature_name].items():
                                demo_dist = group_stats.get('distribution', {})
                                if not demo_dist:
                                    continue
                                
                                demo_dominant = max(demo_dist.items(), key=lambda x: x[1]) if demo_dist else (None, 0)
                                
                                # When dominant values differ between L1 and L2
                                if (baseline_dominant[0] and demo_dominant[0] and 
                                    baseline_dominant[0] != demo_dominant[0]):
                                    
                                    shift_pattern = {
                                        'model': model.upper(),
                                        'object': object_name.title(),
                                        'feature': feature_name.replace('product_', ''),
                                        'demographic': f"{demo_type}_{group_name}",
                                        'baseline_value': baseline_dominant[0],
                                        'baseline_dominance': round(baseline_dominant[1] / sum(baseline_dist.values()), 3),
                                        'demo_value': demo_dominant[0],
                                        'demo_dominance': round(demo_dominant[1] / sum(demo_dist.values()), 3),
                                        'shift_description': f"{baseline_dominant[0]} → {demo_dominant[0]}"
                                    }
                                    shift_patterns.append(shift_pattern)
        
        # Result output
        print(f"\n🔄 Total {len(shift_patterns)} L1→L2 Shifts found:")
        
        # Most dramatic shifts (both high dominance)
        dramatic_shifts = [s for s in shift_patterns if s['baseline_dominance'] >= 0.7 and s['demo_dominance'] >= 0.7]
        
        print(f"\n⚡ Dramatic Shifts ({len(dramatic_shifts)} - both dominance ≥ 70%):")
        for i, shift in enumerate(dramatic_shifts[:10], 1):
            print(f"   {i}. {shift['model']} {shift['object']} {shift['feature']}: "
                  f"{shift['shift_description']} "
                  f"({shift['demographic']})")
        
        return shift_patterns, dramatic_shifts
    
    def save_data(self, results, pivot_df, perfect_cases=None, shift_patterns=None):
        """Save data"""
        # Use comprehensive_analysis directory - always use outputs/comprehensive_analysis
        output_dir = Path("outputs/comprehensive_analysis/vac_scores")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pivot table (main result)
        pivot_df.to_csv(output_dir / "vac_matrix.csv")
        
        # Perfect segregation cases (for reproducibility)
        if perfect_cases is not None and len(perfect_cases) > 0:
            pd.DataFrame(perfect_cases).to_csv(output_dir / "perfect_segregation_cases.csv", index=False)
        
        # Dramatic shifts (for reproducibility)
        if shift_patterns is not None and len(shift_patterns) > 0:
            _, dramatic_shifts_df = shift_patterns
            pd.DataFrame(dramatic_shifts_df).to_csv(output_dir / "dramatic_shifts.csv", index=False)
        
        print(f"\n💾 Data saved: {output_dir}")
        print(f"📄 vac_matrix.csv - VAC score matrix (main result)")
        if perfect_cases is not None and len(perfect_cases) > 0:
            print(f"📄 perfect_segregation_cases.csv - Perfect segregation cases (reproducibility)")
        if shift_patterns is not None and len(shift_patterns) > 0:
            print(f"📄 dramatic_shifts.csv - Dramatic shift cases (reproducibility)")
        
        # Output pivot table to console
        print(f"\n📊 VAC Score Matrix:")
        print("="*60)
        print(pivot_df.round(3))
        
        return pivot_df
    
    def run_extraction(self):
        """Execute complete extraction process"""
        print("🚀 VAC Score Data Extraction")
        
        if not self.load_analysis_results():
            print("❌ Data loading failed")
            return False
        
        # Analyze feature distribution first
        self.analyze_feature_distribution()
            
        results = self.extract_model_bias_data()
        pivot_df = self.create_pivot_table(results)
        
        # Perfect segregation analysis
        perfect_cases, model_perfect_counts = self.find_perfect_segregation_cases()
        
        # Shift patterns analysis
        shift_patterns = self.analyze_baseline_vs_demographic_shifts()
        
        # Save results
        final_df = self.save_data(results, pivot_df, perfect_cases, shift_patterns)
        
        print(f"\n🎉 Data extraction completed!")
        print(f"📋 VAC Score analysis completed")
        print(f"   - VAC score calculation based on product features")
        print(f"   - Including color features")
        print(f"   - Model-specific VAC pattern analysis")
        print(f"   - Perfect segregation case analysis")
        print(f"   - Baseline vs Demographic shifts analysis")
        
        return True

# Execution
if __name__ == "__main__":
    extractor = ModelCentricDataExtractor()
    result = extractor.run_extraction()