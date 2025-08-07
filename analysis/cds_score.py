#!/usr/bin/env python3
"""
SODA Cross-Demographic Diversity Score Analysis
Calculates cross-demographic diversity using JS divergence
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import os

class CrossDemographicDiversityAnalyzer:
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
                
                # Load comprehensive bias analysis
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
                else:
                    print(f"⚠️ Bias analysis file not found: {folder_name}")
        
        print(f"\n📊 Total {loaded_count} combinations loaded")
        return loaded_count > 0

    def calculate_js_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        if not isinstance(p, dict) or not isinstance(q, dict):
            return 0.0
        
        # Normalize distributions
        p_sum = sum(p.values()) if p.values() else 0
        q_sum = sum(q.values()) if q.values() else 0
        
        if p_sum == 0 or q_sum == 0:
            return 0.0
        
        p_norm = {k: v/p_sum for k, v in p.items()}
        q_norm = {k: v/q_sum for k, v in q.items()}
        
        # Get all unique keys
        all_keys = set(p_norm.keys()) | set(q_norm.keys())
        
        if not all_keys:
            return 0.0
        
        # Calculate m = 0.5 * (p + q)
        m = {}
        for key in all_keys:
            p_val = p_norm.get(key, 0)
            q_val = q_norm.get(key, 0)
            m[key] = 0.5 * (p_val + q_val)
        
        # Calculate KL divergence KL(p||m)
        kl_pm = 0.0
        for key in all_keys:
            p_val = p_norm.get(key, 0)
            m_val = m[key]
            if p_val > 0 and m_val > 0:
                kl_pm += p_val * np.log2(p_val / m_val)
        
        # Calculate KL divergence KL(q||m)
        kl_qm = 0.0
        for key in all_keys:
            q_val = q_norm.get(key, 0)
            m_val = m[key]
            if q_val > 0 and m_val > 0:
                kl_qm += q_val * np.log2(q_val / m_val)
        
        # JS divergence = 0.5 * (KL(p||m) + KL(q||m))
        js_divergence = 0.5 * (kl_pm + kl_qm)
        return js_divergence

    def calculate_cds_for_dimension_and_attribute(self, group_distributions):
        """Calculate CDS for a specific demographic dimension and attribute"""
        if len(group_distributions) < 2:
            return 0.0
        
        # Get all unordered pairs of demographic groups
        groups = list(group_distributions.keys())
        pairs = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                pairs.append((groups[i], groups[j]))
        
        if not pairs:
            return 0.0
        
        # Calculate JS divergences for all pairs
        js_divergences = []
        for g1, g2 in pairs:
            p = group_distributions[g1]
            q = group_distributions[g2]
            
            # Ensure we have valid distributions
            if not isinstance(p, dict) or not isinstance(q, dict):
                continue
                
            js_div = self.calculate_js_divergence(p, q)
            js_divergences.append(js_div)
        
        # CDS = average of all JS divergences for this attribute
        return np.mean(js_divergences) if js_divergences else 0.0

    def aggregate_attribute_data_across_all_combinations(self):
        """Aggregate attribute distributions across all model-object combinations"""
        print("\n" + "="*60)
        print("📊 Aggregating attribute data across all combinations...")
        print("="*60)
        
        # Structure: {attribute_name: {dimension: {group: aggregated_distribution}}}
        aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        
        # Collect all data
        for folder_name, data in self.results.items():
            model = data['model']
            object_name = data['object']
            bias_analysis = data['bias_analysis']
            demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
            
            print(f"  🔍 Processing {folder_name}...")
            
            # Process each demographic dimension
            for dimension in ['age', 'gender', 'ethnicity']:
                if dimension not in demographic_stats:
                    continue
                    
                # Process each attribute in this dimension
                for attribute_name, groups_data in demographic_stats[dimension].items():
                    # Clean attribute name
                    clean_attr_name = attribute_name
                    if attribute_name.startswith('product_product_'):
                        clean_attr_name = attribute_name.replace('product_product_', '')
                    elif attribute_name.startswith('product_'):
                        clean_attr_name = attribute_name.replace('product_', '')
                    elif attribute_name.startswith('background_'):
                        clean_attr_name = attribute_name.replace('background_', '')
                    
                    # Process each group in this attribute
                    for group_name, group_stats in groups_data.items():
                        if group_name not in self.demographic_groups:
                            continue
                            
                        # Get distribution
                        distribution = group_stats.get('distribution', group_stats)
                        if isinstance(distribution, dict):
                            # Aggregate the distributions
                            for value, count in distribution.items():
                                aggregated_data[clean_attr_name][dimension][group_name][value] += count
        
        return aggregated_data

    def create_paper_table(self, aggregated_data):
        """Create the exact table structure from the paper"""
        print("\n" + "="*60)
        print("📊 Creating paper-aligned table...")
        print("="*60)
        
        table_data = []
        
        # Calculate CDS for each attribute and dimension combination
        for attribute_name in sorted(aggregated_data.keys()):
            print(f"  📊 Processing attribute: {attribute_name}")
            
            attr_data = aggregated_data[attribute_name]
            
            # Calculate CDS for each dimension
            age_cds = 0.0
            gender_cds = 0.0 
            ethnicity_cds = 0.0
            
            # Age CDS
            if 'age' in attr_data and len(attr_data['age']) >= 2:
                age_group_distributions = dict(attr_data['age'])
                age_cds = self.calculate_cds_for_dimension_and_attribute(age_group_distributions)
            
            # Gender CDS
            if 'gender' in attr_data and len(attr_data['gender']) >= 2:
                gender_group_distributions = dict(attr_data['gender'])
                gender_cds = self.calculate_cds_for_dimension_and_attribute(gender_group_distributions)
            
            # Ethnicity CDS
            if 'ethnicity' in attr_data and len(attr_data['ethnicity']) >= 2:
                ethnicity_group_distributions = dict(attr_data['ethnicity'])
                ethnicity_cds = self.calculate_cds_for_dimension_and_attribute(ethnicity_group_distributions)
            
            # Determine which objects this attribute appears in
            objects_with_attr = set()
            for folder_name, data in self.results.items():
                bias_analysis = data['bias_analysis']
                demographic_stats = bias_analysis.get('demographic_bias_analysis', {})
                for dimension in ['age', 'gender', 'ethnicity']:
                    if dimension in demographic_stats:
                        # Check both original and cleaned attribute names
                        original_attrs = set(demographic_stats[dimension].keys())
                        for orig_attr in original_attrs:
                            clean_attr = orig_attr
                            if orig_attr.startswith('product_product_'):
                                clean_attr = orig_attr.replace('product_product_', '')
                            elif orig_attr.startswith('product_'):
                                clean_attr = orig_attr.replace('product_', '')
                            elif orig_attr.startswith('background_'):
                                clean_attr = orig_attr.replace('background_', '')
                            
                            if clean_attr == attribute_name:
                                objects_with_attr.add(data['object'].title())
            
            # Create objects string
            if len(objects_with_attr) == 5:  # All objects
                objects_str = "All Objects"
            elif len(objects_with_attr) == 1:
                objects_str = list(objects_with_attr)[0]
            else:
                objects_str = ", ".join(sorted(objects_with_attr))
            
            table_data.append({
                'Attribute': attribute_name.replace('_', ' ').title(),
                'Objects': objects_str,
                'Age': round(age_cds, 3),
                'Gender': round(gender_cds, 3),
                'Ethnicity': round(ethnicity_cds, 3)
            })
            
            print(f"    ✅ Age: {age_cds:.3f}, Gender: {gender_cds:.3f}, Ethnicity: {ethnicity_cds:.3f}")
        
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by the sum of all three dimensions (highest diversity first)
        df['Total_Score'] = df['Age'] + df['Gender'] + df['Ethnicity']
        df = df.sort_values('Total_Score', ascending=False)
        
        # Remove the temporary Total_Score column
        df = df.drop('Total_Score', axis=1)
        
        return df

    def run_analysis(self):
        """Run complete paper-aligned CDS analysis"""
        print("🚀 Paper-Aligned Cross-Demographic Diversity Score (CDS) Analysis")
        print("="*60)
    
        # 1. Load data
        if not self.load_analysis_results():
            print("❌ Data loading failed")
            return False
        
        # 2. Aggregate attribute data across all combinations
        aggregated_data = self.aggregate_attribute_data_across_all_combinations()
        
        if not aggregated_data:
            print("❌ No aggregated data generated")
            return False
        
        # 3. Create paper-aligned table
        paper_table = self.create_paper_table(aggregated_data)
        
        if paper_table is None or paper_table.empty:
            print("❌ Paper table creation failed")
            return False
        
        # 4. Save table
        # Use comprehensive_analysis directory - always use outputs/comprehensive_analysis
        output_dir = Path("outputs/comprehensive_analysis/cds_score")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paper_table.to_csv(output_dir / "cds_feature_diversity_matrix.csv", index=False)
        
        print(f"\n💾 Saved file:")
        print(f"   📄 cds_score/cds_feature_diversity_matrix.csv - CDS feature diversity matrix")
        
        # 5. Display results
        print(f"\n📋 CDS Feature Diversity Matrix (Top 10):")
        print("="*80)
        print(paper_table.head(10).to_string(index=False))
    
        # 6. Summary statistics
        print(f"\n📊 Summary Statistics:")
        print("="*40)
        print(f"Total attributes analyzed: {len(paper_table)}")
        print(f"Average Age CDS: {paper_table['Age'].mean():.3f}")
        print(f"Average Gender CDS: {paper_table['Gender'].mean():.3f}")
        print(f"Average Ethnicity CDS: {paper_table['Ethnicity'].mean():.3f}")
        
        # Find highest scoring attributes per dimension
        print(f"\n🏆 Highest CDS by Dimension:")
        print("="*40)
        max_age = paper_table.loc[paper_table['Age'].idxmax()]
        max_gender = paper_table.loc[paper_table['Gender'].idxmax()]
        max_ethnicity = paper_table.loc[paper_table['Ethnicity'].idxmax()]
        
        print(f"Age: {max_age['Attribute']} ({max_age['Age']:.3f})")
        print(f"Gender: {max_gender['Attribute']} ({max_gender['Gender']:.3f})")
        print(f"Ethnicity: {max_ethnicity['Attribute']} ({max_ethnicity['Ethnicity']:.3f})")
        
        return True

# Execute
if __name__ == "__main__":
    analyzer = CrossDemographicDiversityAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n✅ CDS analysis completed!")
        print("\n💡 Usage tips:")
        print("   - Use cds_score/cds_feature_diversity_matrix.csv as main table")
        print("   - Select top 8-10 attributes for the final paper table")
        print("   - Bold the highest values in each column")
    else:
        print("\n❌ CDS analysis failed")