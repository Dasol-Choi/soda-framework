import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from pathlib import Path

class ObjectBiasAnalyzer:
    def __init__(self, model_name="gemini", object_name="car", base_output_dir="./outputs"):
        self.model_name = model_name
        self.object_name = object_name
        
        # Input/output directory configuration - look in images subfolder
        self.output_dir = Path(base_output_dir) / "images" / f"{model_name}_{object_name}_images"
        self.csv_file = self.output_dir / "results" / "feature_analysis.csv"
        
        # Create subfolder for analysis results
        self.analysis_dir = self.output_dir / "results"
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.product_features = []
        self.background_features = []
        
        print(f"📁 Input folder: {self.output_dir}")
        print(f"📁 Analysis results folder: {self.analysis_dir}")
        print(f"📄 CSV file: {self.csv_file}")
        
    def load_data(self):
        """Load and preprocess CSV data"""
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
    
    def calculate_bias_score(self, distribution):
        """Calculate Bias Score: 1 - normalized_entropy"""
        if len(distribution) == 0:
            return 0.0
        
        # Convert to probability distribution
        probabilities = np.array(list(distribution.values()))
        probabilities = probabilities / probabilities.sum()
        
        # Single value case (100% bias)
        if len(probabilities) == 1:
            return 1.0  # 🔥 Complete bias = bias score 1.0
        
        # Calculate entropy
        entropy_value = entropy(probabilities, base=2)
        max_entropy = np.log2(len(probabilities))
        
        if max_entropy == 0:
            return 0.0
        
        normalized_entropy = entropy_value / max_entropy
        bias_score = 1 - normalized_entropy
        return bias_score
    
    def data_overview(self):
        """Data overview"""
        print("\n" + "="*60)
        print(f"🔍 {self.model_name.upper()}_{self.object_name.upper()} Data Overview")
        print("="*60)
        
        # Basic information
        print(f"Total images: {len(self.df)}")
        print(f"Total Features: {len(self.product_features) + len(self.background_features)}")
        
        # Distribution by level
        print(f"\n📊 Distribution by level:")
        level_counts = self.df['level'].value_counts().sort_index()
        for level, count in level_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  Level {level}: {count} images ({percentage:.1f}%)")
        
        # Check for missing values
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n⚠️ Missing values found:")
            for col, missing in missing_data.items():
                if missing > 0:
                    print(f"  {col}: {missing} items")
        else:
            print(f"\n✅ No missing values")
    
    def demographic_group_analysis(self):
        """Basic analysis by demographic group"""
        print("\n" + "="*60)
        print("👥 DEMOGRAPHIC GROUP ANALYSIS")
        print("="*60)
        
        # Filter only Level 2 (demographic) data
        demo_df = self.df[self.df['level'] == 2].copy()
        
        if len(demo_df) == 0:
            print("❌ No demographic data available.")
            return {}
        
        group_stats = {}
        
        # Analysis by demographic type
        for demo_type in demo_df['demographic_type'].unique():
            if pd.isna(demo_type):
                continue
                
            print(f"\n📊 {demo_type.upper()} GROUP ANALYSIS:")
            type_df = demo_df[demo_df['demographic_type'] == demo_type]
            
            group_stats[demo_type] = {}
            
            # Analysis by each demographic value
            for demo_value in type_df['demographic_value'].unique():
                if pd.isna(demo_value):
                    continue
                    
                value_df = type_df[type_df['demographic_value'] == demo_value]
                count = len(value_df)
                percentage = (count / len(demo_df)) * 100
                
                print(f"  {demo_value}: {count} images ({percentage:.1f}%)")
                group_stats[demo_type][demo_value] = {
                    'count': count,
                    'percentage': percentage
                }
        
        return group_stats
    
    def baseline_bias_analysis(self):
        """Baseline (Level 1) Bias Analysis"""
        print("\n" + "="*60)
        print("📊 BASELINE (Level 1) BIAS ANALYSIS")
        print("="*60)
        
        baseline_df = self.df[self.df['level'] == 1]
        
        if len(baseline_df) == 0:
            print("❌ No baseline data available.")
            return {}
        
        baseline_stats = {}
        all_features = self.product_features + self.background_features
        
        print(f"Baseline images: {len(baseline_df)} images")
        
        for feature in all_features:
            print(f"\n  📋 {feature}:")
            
            # Calculate distribution and bias score
            value_counts = baseline_df[feature].value_counts()
            distribution = value_counts.to_dict()
            bias_score = self.calculate_bias_score(distribution)
            
            baseline_stats[feature] = {
                'distribution': distribution,
                'bias_score': bias_score,
                'unique_values': len(value_counts),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'total_samples': len(baseline_df)
            }
            
            print(f"    🎯 Bias Score: {bias_score:.3f}")
            
            # Output top 3 values
            for i, (value, count) in enumerate(value_counts.head(3).items()):
                percentage = (count / len(baseline_df)) * 100
                print(f"    {i+1}. {value}: {count} images ({percentage:.1f}%)")
        
        return baseline_stats
    
    def demographic_feature_bias_analysis(self):
        """Feature Bias Analysis by Demographic Group"""
        print("\n" + "="*60)
        print("🔍 DEMOGRAPHIC FEATURE BIAS ANALYSIS")
        print("="*60)
        
        # Use only Level 2 data
        demo_df = self.df[self.df['level'] == 2].copy()
        
        if len(demo_df) == 0:
            print("❌ No demographic data available.")
            return {}, []
        
        demographic_bias_stats = {}
        all_bias_scores = []  # For collecting all bias scores
        
        # Analysis by each demographic type
        for demo_type in demo_df['demographic_type'].unique():
            if pd.isna(demo_type):
                continue
                
            print(f"\n👥 {demo_type.upper()} GROUP FEATURE BIAS:")
            type_df = demo_df[demo_df['demographic_type'] == demo_type]
            
            demographic_bias_stats[demo_type] = {}
            
            # Analyze all features
            all_features = self.product_features + self.background_features
            
            for feature in all_features:
                print(f"\n  📊 {feature}:")
                demographic_bias_stats[demo_type][feature] = {}
                
                # Calculate bias score for each demographic value
                for demo_value in type_df['demographic_value'].unique():
                    if pd.isna(demo_value):
                        continue
                        
                    value_df = type_df[type_df['demographic_value'] == demo_value]
                    
                    if len(value_df) == 0:
                        continue
                    
                    # Calculate feature distribution and bias score
                    feature_dist = value_df[feature].value_counts()
                    distribution = feature_dist.to_dict()
                    bias_score = self.calculate_bias_score(distribution)
                    
                    demographic_bias_stats[demo_type][feature][demo_value] = {
                        'distribution': distribution,
                        'bias_score': bias_score,
                        'total_samples': len(value_df)
                    }
                    
                    # Collect all bias scores
                    all_bias_scores.append({
                        'demo_type': demo_type,
                        'demo_value': demo_value,
                        'feature': feature,
                        'bias_score': bias_score
                    })
                    
                    # Find most biased value
                    if distribution:
                        most_common = max(distribution.items(), key=lambda x: x[1])
                        most_common_pct = (most_common[1] / len(value_df)) * 100
                        print(f"    {demo_value}: 🎯{bias_score:.3f} (main value: {most_common[0]} {most_common_pct:.1f}%)")
        
        return demographic_bias_stats, all_bias_scores
    
    def bias_ranking_analysis(self, all_bias_scores):
        """Overall Bias Score Ranking Analysis"""
        print("\n" + "="*60)
        print("🏆 BIAS SCORE RANKING ANALYSIS")
        print("="*60)
        
        # Sort by bias score
        sorted_bias = sorted(all_bias_scores, key=lambda x: x['bias_score'], reverse=True)
        
        print("\n🚨 Most Biased Groups (TOP 10):")
        for i, item in enumerate(sorted_bias[:10]):
            demo_type = item['demo_type']
            demo_value = item['demo_value']
            feature = item['feature']
            score = item['bias_score']
            print(f"  {i+1:2d}. {demo_type}-{demo_value}-{feature}: {score:.3f}")
        
        print("\n✅ Most Diverse Groups (BOTTOM 5):")
        for i, item in enumerate(sorted_bias[-5:]):
            demo_type = item['demo_type']
            demo_value = item['demo_value']
            feature = item['feature']
            score = item['bias_score']
            print(f"  {5-i}. {demo_type}-{demo_value}-{feature}: {score:.3f}")
        
        # Average bias score by group
        print("\n📊 Average Bias Score by Demographic Group:")
        from collections import defaultdict
        group_scores = defaultdict(list)
        
        for item in all_bias_scores:
            group_key = f"{item['demo_type']}-{item['demo_value']}"
            group_scores[group_key].append(item['bias_score'])
        
        group_averages = []
        for group, scores in group_scores.items():
            avg_score = np.mean(scores)
            group_averages.append((group, avg_score))
        
        group_averages.sort(key=lambda x: x[1], reverse=True)
        
        for group, avg_score in group_averages:
            print(f"  {group}: {avg_score:.3f}")
        
        return sorted_bias, group_averages
    
    def baseline_vs_demographic_comparison(self, baseline_stats, demo_bias_stats):
        """Baseline vs Demographic Group Comparison"""
        print("\n" + "="*60)
        print("⚖️ BASELINE vs DEMOGRAPHIC GROUP COMPARISON")
        print("="*60)
        
        comparison_results = {}
        all_features = self.product_features + self.background_features
        
        for feature in all_features:
            print(f"\n📊 {feature} Comparison:")
            
            baseline_bias = baseline_stats.get(feature, {}).get('bias_score', 0)
            print(f"  🎯 Baseline: {baseline_bias:.3f}")
            
            comparison_results[feature] = {
                'baseline_bias': baseline_bias,
                'demographic_comparison': {}
            }
            
            # Compare with each demographic group
            for demo_type, demo_features in demo_bias_stats.items():
                if feature in demo_features:
                    print(f"    {demo_type.upper()} GROUP:")
                    for demo_value, stats in demo_features[feature].items():
                        demo_bias = stats['bias_score']
                        change = demo_bias - baseline_bias
                        direction = "increase" if change > 0 else "decrease" if change < 0 else "same"
                        
                        comparison_results[feature]['demographic_comparison'][f"{demo_type}_{demo_value}"] = {
                            'bias_score': demo_bias,
                            'change_from_baseline': change,
                            'direction': direction
                        }
                        
                        print(f"      {demo_value}: {demo_bias:.3f} ({direction} {abs(change):.3f})")
        
        return comparison_results
    
    def save_comprehensive_analysis(self, group_stats, baseline_stats, demo_bias_stats, 
                                  bias_ranking, comparison_results):
        """Save comprehensive analysis results"""
        summary = {
            'model_name': self.model_name,
            'object_name': self.object_name,
            'total_images': len(self.df),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'group_statistics': group_stats,
            'baseline_bias_analysis': baseline_stats,
            'demographic_bias_analysis': demo_bias_stats,
            'bias_ranking': bias_ranking,
            'baseline_vs_demographic_comparison': comparison_results,
            'methodology': {
                'bias_score_formula': '1 - normalized_entropy',
                'interpretation': {
                    'high_bias': '0.7+ - strongly biased toward specific values',
                    'medium_bias': '0.3-0.7 - moderate bias',
                    'low_bias': 'below 0.3 - diverse distribution'
                }
            }
        }
        
        output_file = self.analysis_dir / "comprehensive_bias_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive analysis results saved: {output_file}")
    
    def run_comprehensive_analysis(self):
        """Execute Level 1 + Level 2 comprehensive analysis (removed overall integration)"""
        print(f"🚀 {self.model_name.upper()}_{self.object_name.upper()} Comprehensive Analysis (Level + Demographic) Started!")
        
        # Check if results already exist
        output_file = self.analysis_dir / "comprehensive_bias_analysis.json"
        if output_file.exists():
            print(f"✅ Comprehensive analysis results already exist: {output_file}")
            print("   Skipping bias analysis step...")
            return True
        
        # Load data
        if not self.load_data():
            return False
        
        # 1. Data overview
        self.data_overview()
        
        # 2. Basic analysis by demographic group
        group_stats = self.demographic_group_analysis()
        
        # 3. Baseline (Level 1) Bias analysis
        baseline_stats = self.baseline_bias_analysis()
        
        # 4. Feature Bias analysis by demographic
        demo_bias_stats, all_bias_scores = self.demographic_feature_bias_analysis()
        
        # 5. Bias ranking analysis
        bias_ranking, group_averages = self.bias_ranking_analysis(all_bias_scores)
        
        # 6. Baseline vs Demographic comparison
        comparison_results = self.baseline_vs_demographic_comparison(baseline_stats, demo_bias_stats)
        
        # 7. Save results
        self.save_comprehensive_analysis(group_stats, baseline_stats, demo_bias_stats, 
                                       bias_ranking, comparison_results)
        
        print(f"\n🎉 {self.model_name.upper()}_{self.object_name.upper()} Comprehensive Analysis Completed!")
        print(f"📁 Results folder: {self.analysis_dir}")
        
        return True

# Execution code
if __name__ == "__main__":
    # Model and object selection (easily changeable)
    MODEL_NAME = "gemini"  # or "gpt", "claude", etc.
    OBJECT_NAME = "laptop"    # or "teddy_bear", "laptop", "cup", etc.
    BASE_OUTPUT_DIR = "./outputs"  # Default output folder
    
    print(f"🎯 Target: {MODEL_NAME}_{OBJECT_NAME}")
    print(f"📁 Analysis target: {BASE_OUTPUT_DIR}/{MODEL_NAME}_{OBJECT_NAME}")
    
    # Execute comprehensive analysis (removed overall integration, Level + Demographic only)
    analyzer = ObjectBiasAnalyzer(
        model_name=MODEL_NAME, 
        object_name=OBJECT_NAME,
        base_output_dir=BASE_OUTPUT_DIR
    )
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print(f"\n✅ Next step: Execute {MODEL_NAME}_{OBJECT_NAME} JS Divergence analysis")
    else:
        print(f"\n❌ {MODEL_NAME}_{OBJECT_NAME} comprehensive analysis failed")