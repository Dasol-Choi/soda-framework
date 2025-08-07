import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import itertools
import json
from pathlib import Path
from collections import Counter

class ObjectBiasJSDivergenceAnalyzer:
    def __init__(self, model_name="gemini", object_name="car", base_output_dir="./outputs"):
        self.model_name = model_name
        self.object_name = object_name
        
        # Input/output directory configuration - look in images subfolder
        self.output_dir = Path(base_output_dir) / "images" / f"{model_name}_{object_name}_images"
        self.csv_file = self.output_dir / "results" / "feature_analysis.csv"
        
        # Use subfolder for analysis results (reuse existing results folder)
        self.analysis_dir = self.output_dir / "results"
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.product_features = []
        self.background_features = []
        
        print(f"📁 Input folder: {self.output_dir}")
        print(f"📁 Analysis results folder: {self.analysis_dir}")
        print(f"📄 CSV file: {self.csv_file}")
        
    def load_data(self):
        """Load CSV data"""
        try:
            if not self.csv_file.exists():
                print(f"❌ CSV file does not exist: {self.csv_file}")
                print("   Please run Feature Extraction first!")
                return False
                
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Data loading completed: {len(self.df)} images")
            
            # Feature column classification
            self.product_features = [col for col in self.df.columns if col.startswith('product_')]
            self.background_features = [col for col in self.df.columns if col.startswith('background_')]
            
            print(f"📊 Product Features: {len(self.product_features)}")
            print(f"🖼️ Background Features: {len(self.background_features)}")
            
            return True
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def get_probability_distribution(self, data, feature):
        """Calculate probability distribution for a specific group's feature"""
        if len(data) == 0:
            return {}
        
        value_counts = data[feature].value_counts()
        total = len(data)
        
        # Convert to probability distribution
        prob_dist = {value: count / total for value, count in value_counts.items()}
        return prob_dist
    
    def align_distributions(self, dist1, dist2):
        """Align two distributions with the same keys (missing values treated as 0)"""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        aligned_dist1 = [dist1.get(key, 0) for key in sorted(all_keys)]
        aligned_dist2 = [dist2.get(key, 0) for key in sorted(all_keys)]
        
        return np.array(aligned_dist1), np.array(aligned_dist2), sorted(all_keys)
    
    def calculate_js_divergence(self, dist1, dist2):
        """Calculate Jensen-Shannon Divergence"""
        if not dist1 or not dist2:
            return 0.0
        
        # Align distributions
        aligned_dist1, aligned_dist2, _ = self.align_distributions(dist1, dist2)
        
        # Handle zero probability (replace with very small value)
        epsilon = 1e-10
        aligned_dist1 = np.maximum(aligned_dist1, epsilon)
        aligned_dist2 = np.maximum(aligned_dist2, epsilon)
        
        # Normalize
        aligned_dist1 = aligned_dist1 / aligned_dist1.sum()
        aligned_dist2 = aligned_dist2 / aligned_dist2.sum()
        
        # Calculate JS Divergence
        js_distance = jensenshannon(aligned_dist1, aligned_dist2)
        
        return js_distance
    
    def baseline_vs_demographic_divergence(self):
        """Baseline vs each Demographic group JS Divergence"""
        print("\n" + "="*60)
        print("📊 BASELINE vs DEMOGRAPHIC JS DIVERGENCE")
        print("="*60)
        
        baseline_df = self.df[self.df['level'] == 1]
        demo_df = self.df[self.df['level'] == 2]
        
        if len(baseline_df) == 0 or len(demo_df) == 0:
            print("❌ No Baseline or Demographic data available.")
            return {}
        
        baseline_divergence_results = {}
        all_features = self.product_features + self.background_features
        
        print(f"Baseline: {len(baseline_df)} vs Demographic: {len(demo_df)}")
        
        # Compare baseline with each demographic group
        for demo_type in demo_df['demographic_type'].unique():
            if pd.isna(demo_type):
                continue
            
            print(f"\n👥 {demo_type.upper()} group vs Baseline:")
            type_df = demo_df[demo_df['demographic_type'] == demo_type]
            
            baseline_divergence_results[demo_type] = {}
            
            for demo_value in type_df['demographic_value'].unique():
                if pd.isna(demo_value):
                    continue
                
                value_df = type_df[type_df['demographic_value'] == demo_value]
                
                print(f"\n  📊 {demo_value} vs Baseline:")
                baseline_divergence_results[demo_type][demo_value] = {}
                
                # Calculate JS Divergence for each feature
                for feature in all_features:
                    baseline_dist = self.get_probability_distribution(baseline_df, feature)
                    demo_dist = self.get_probability_distribution(value_df, feature)
                    
                    js_divergence = self.calculate_js_divergence(baseline_dist, demo_dist)
                    
                    baseline_divergence_results[demo_type][demo_value][feature] = {
                        'js_divergence': js_divergence,
                        'baseline_samples': len(baseline_df),
                        'demographic_samples': len(value_df)
                    }
                    
                    # Add interpretation
                    if js_divergence < 0.1:
                        interpretation = "Very Similar"
                    elif js_divergence < 0.3:
                        interpretation = "Slightly Different"
                    elif js_divergence < 0.5:
                        interpretation = "Moderately Different"
                    elif js_divergence < 0.7:
                        interpretation = "Considerably Different"
                    else:
                        interpretation = "Very Different"
                    
                    print(f"    {feature}: {js_divergence:.3f} ({interpretation})")
        
        return baseline_divergence_results
    
    def cross_demographic_divergence(self):
        """Cross-demographic JS Divergence (within Age, Gender, Ethnicity)"""
        print("\n" + "="*60)
        print("🔄 CROSS-DEMOGRAPHIC JS DIVERGENCE")
        print("="*60)
        
        demo_df = self.df[self.df['level'] == 2]
        
        if len(demo_df) == 0:
            print("❌ No Demographic data available.")
            return {}
        
        cross_divergence_results = {}
        all_features = self.product_features + self.background_features
        
        # Compare groups within each demographic type
        for demo_type in demo_df['demographic_type'].unique():
            if pd.isna(demo_type):
                continue
            
            print(f"\n👥 {demo_type.upper()} group internal comparison:")
            type_df = demo_df[demo_df['demographic_type'] == demo_type]
            demo_values = [val for val in type_df['demographic_value'].unique() if not pd.isna(val)]
            
            if len(demo_values) < 2:
                print(f"  ⚠️ Insufficient groups to compare in {demo_type}.")
                continue
            
            cross_divergence_results[demo_type] = {}
            
            # Compare all possible combinations
            for val1, val2 in itertools.combinations(demo_values, 2):
                comparison_key = f"{val1}_vs_{val2}"
                print(f"\n  📊 {val1} vs {val2}:")
                
                df1 = type_df[type_df['demographic_value'] == val1]
                df2 = type_df[type_df['demographic_value'] == val2]
                
                cross_divergence_results[demo_type][comparison_key] = {}
                
                # Calculate JS Divergence for each feature
                for feature in all_features:
                    dist1 = self.get_probability_distribution(df1, feature)
                    dist2 = self.get_probability_distribution(df2, feature)
                    
                    js_divergence = self.calculate_js_divergence(dist1, dist2)
                    
                    cross_divergence_results[demo_type][comparison_key][feature] = {
                        'js_divergence': js_divergence,
                        'group1_samples': len(df1),
                        'group2_samples': len(df2)
                    }
                    
                    # Add interpretation
                    if js_divergence < 0.1:
                        interpretation = "Very Similar"
                    elif js_divergence < 0.3:
                        interpretation = "Slightly Different"
                    elif js_divergence < 0.5:
                        interpretation = "Moderately Different"
                    elif js_divergence < 0.7:
                        interpretation = "Considerably Different"
                    else:
                        interpretation = "Very Different"
                    
                    print(f"    {feature}: {js_divergence:.3f} ({interpretation})")
        
        return cross_divergence_results
    
    def divergence_ranking_analysis(self, baseline_results, cross_results):
        """JS Divergence ranking analysis"""
        print("\n" + "="*60)
        print("🏆 JS DIVERGENCE RANKING ANALYSIS")
        print("="*60)
        
        all_divergences = []
        
        # Collect Baseline vs Demographic divergences
        for demo_type, demo_values in baseline_results.items():
            for demo_value, features in demo_values.items():
                for feature, result in features.items():
                    all_divergences.append({
                        'comparison_type': 'baseline_vs_demographic',
                        'comparison': f"baseline_vs_{demo_type}_{demo_value}",
                        'feature': feature,
                        'js_divergence': result['js_divergence']
                    })
        
        # Collect Cross-demographic divergences
        for demo_type, comparisons in cross_results.items():
            for comparison, features in comparisons.items():
                for feature, result in features.items():
                    all_divergences.append({
                        'comparison_type': 'cross_demographic',
                        'comparison': f"{demo_type}_{comparison}",
                        'feature': feature,
                        'js_divergence': result['js_divergence']
                    })
        
        # Sort by JS Divergence
        sorted_divergences = sorted(all_divergences, key=lambda x: x['js_divergence'], reverse=True)
        
        print("\n🚨 Comparisons showing the biggest differences (TOP 15):")
        for i, item in enumerate(sorted_divergences[:15]):
            comparison = item['comparison']
            feature = item['feature']
            divergence = item['js_divergence']
            comp_type = item['comparison_type']
            print(f"  {i+1:2d}. {comparison} - {feature}: {divergence:.3f} ({comp_type})")
        
        print("\n✅ Most similar comparisons (BOTTOM 5):")
        for i, item in enumerate(sorted_divergences[-5:]):
            comparison = item['comparison']
            feature = item['feature']
            divergence = item['js_divergence']
            comp_type = item['comparison_type']
            print(f"  {5-i}. {comparison} - {feature}: {divergence:.3f} ({comp_type})")
        
        return sorted_divergences
    
    def feature_divergence_summary(self, baseline_results, cross_results):
        """Feature-wise average Divergence summary"""
        print("\n" + "="*60)
        print("📈 FEATURE-WISE AVERAGE DIVERGENCE SUMMARY")
        print("="*60)
        
        from collections import defaultdict
        feature_divergences = defaultdict(list)
        
        # Collect all divergences
        for demo_type, demo_values in baseline_results.items():
            for demo_value, features in demo_values.items():
                for feature, result in features.items():
                    feature_divergences[feature].append(result['js_divergence'])
        
        for demo_type, comparisons in cross_results.items():
            for comparison, features in comparisons.items():
                for feature, result in features.items():
                    feature_divergences[feature].append(result['js_divergence'])
        
        # Calculate average by feature
        feature_summary = []
        for feature, divergences in feature_divergences.items():
            avg_divergence = np.mean(divergences)
            max_divergence = np.max(divergences)
            min_divergence = np.min(divergences)
            
            feature_summary.append({
                'feature': feature,
                'avg_divergence': avg_divergence,
                'max_divergence': max_divergence,
                'min_divergence': min_divergence,
                'total_comparisons': len(divergences)
            })
        
        # Sort by average divergence
        feature_summary.sort(key=lambda x: x['avg_divergence'], reverse=True)
        
        print("\n📊 Feature-wise average JS Divergence:")
        for item in feature_summary:
            feature = item['feature']
            avg = item['avg_divergence']
            max_div = item['max_divergence']
            min_div = item['min_divergence']
            count = item['total_comparisons']
            
            print(f"  {feature}:")
            print(f"    Average: {avg:.3f}, Max: {max_div:.3f}, Min: {min_div:.3f} ({count} comparisons)")
        
        return feature_summary
    
    def save_divergence_analysis(self, baseline_results, cross_results, ranking, feature_summary):
        """Save JS Divergence analysis results"""
        results = {
            'model_name': self.model_name,
            'object_name': self.object_name,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'baseline_vs_demographic_divergence': baseline_results,
            'cross_demographic_divergence': cross_results,
            'divergence_ranking': ranking,
            'feature_divergence_summary': feature_summary,
            'methodology': {
                'measure': 'Jensen-Shannon Divergence',
                'interpretation': {
                    'very_similar': '< 0.1 - Very similar distribution',
                    'slightly_different': '0.1-0.3 - Slightly different distribution',
                    'moderately_different': '0.3-0.5 - Moderately different',
                    'considerably_different': '0.5-0.7 - Considerably different',
                    'very_different': '> 0.7 - Very different'
                }
            }
        }
        
        output_file = self.analysis_dir / "js_divergence_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 JS Divergence analysis results saved: {output_file}")
    
    def run_divergence_analysis(self):
        """Run complete 3-step JS Divergence analysis"""
        print(f"🚀 {self.model_name.upper()}_{self.object_name.upper()} Step 3: JS Divergence analysis started!")
        
        # Check if results already exist
        output_file = self.analysis_dir / "js_divergence_analysis.json"
        if output_file.exists():
            print(f"✅ JS divergence analysis results already exist: {output_file}")
            print("   Skipping JS divergence analysis step...")
            return True
        
        # Load data
        if not self.load_data():
            return False
        
        # 1. Baseline vs Demographic JS Divergence
        baseline_results = self.baseline_vs_demographic_divergence()
        
        # 2. Cross-demographic JS Divergence
        cross_results = self.cross_demographic_divergence()
        
        # 3. Divergence ranking analysis
        ranking = self.divergence_ranking_analysis(baseline_results, cross_results)
        
        # 4. Feature-wise Divergence summary
        feature_summary = self.feature_divergence_summary(baseline_results, cross_results)
        
        # 5. Save results
        self.save_divergence_analysis(baseline_results, cross_results, ranking, feature_summary)
        
        print(f"\n🎉 {self.model_name.upper()}_{self.object_name.upper()} Step 3 JS Divergence analysis completed!")
        print(f"📁 Results folder: {self.analysis_dir}")
        
        return True

# Execution code
if __name__ == "__main__":
    # Model and object selection (easily changeable)
    MODEL_NAME = "gemini"  # or "gpt", "claude", etc.
    OBJECT_NAME = "laptop"    # or "teddy_bear", "laptop", "cup", etc.
    BASE_OUTPUT_DIR = "./outputs"  # Default output directory
    
    print(f"🎯 Target: {MODEL_NAME}_{OBJECT_NAME}")
    print(f"📁 Analysis target: {BASE_OUTPUT_DIR}/{MODEL_NAME}_{OBJECT_NAME}")
    
    # Run 3-step JS Divergence analysis
    analyzer = ObjectBiasJSDivergenceAnalyzer(
        model_name=MODEL_NAME, 
        object_name=OBJECT_NAME,
        base_output_dir=BASE_OUTPUT_DIR
    )
    success = analyzer.run_divergence_analysis()
    
    if success:
        print(f"\n✅ {MODEL_NAME}_{OBJECT_NAME} all analysis completed!")
        print("🎉 Step 1(Feature Discovery) + Step 2(Feature Extraction) + Step 3(Bias Analysis) + Step 4(JS Divergence) completed")
    else:
        print(f"\n❌ {MODEL_NAME}_{OBJECT_NAME} JS Divergence analysis failed")