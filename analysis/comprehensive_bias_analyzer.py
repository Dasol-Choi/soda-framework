#!/usr/bin/env python3
"""
SODA Comprehensive Bias Analysis Framework
Integrates three bias analysis methods:
1. BDS (Baseline vs Demographics Score) - JS divergence between baseline and demographic groups
2. VAC (Visual Attribute Consistency Score) - Model-centric bias with perfect segregation and shifts
3. CDS (Cross-Demographic Diversity Score) - Cross-demographic diversity using JS divergence

Usage:
    python comprehensive_bias_analyzer.py [--output_dir OUTPUT_DIR]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Import the three analysis modules
from .bds_score import BaselineVsDemographicsAnalyzer
from .vac_score import ModelCentricDataExtractor
from .cds_score import CrossDemographicDiversityAnalyzer

class ComprehensiveBiasAnalyzer:
    def __init__(self, output_base_dir=None, analysis_output_dir=None):
        # Use default paths from configuration if not provided
        from config.paths import PATHS
        self.output_base_dir = Path(output_base_dir or str(PATHS.OUTPUTS_DIR))
        self.analysis_output_dir = Path(analysis_output_dir or str(PATHS.COMPREHENSIVE_ANALYSIS_DIR))
        
        self.analysis_output_dir.mkdir(exist_ok=True)
        
        print("🚀 SODA Comprehensive Bias Analysis Framework")
        print("="*80)
        print(f"📁 Data source: {self.output_base_dir}")
        print(f"📁 Analysis output: {self.analysis_output_dir}")
        print("="*80)
        
        # Initialize analyzers
        self.bds_analyzer = BaselineVsDemographicsAnalyzer(str(self.output_base_dir))
        self.vac_analyzer = ModelCentricDataExtractor(str(self.output_base_dir))
        self.cds_analyzer = CrossDemographicDiversityAnalyzer(str(self.output_base_dir))
        
        # Results storage
        self.results = {
            'bds': None,
            'vac': None,
            'cds': None,
            'summary': None
        }
    
    def run_bds_analysis(self):
        """Run Baseline vs Demographics Score analysis"""
        print("\n" + "="*60)
        print("📊 1. Running BDS (Baseline vs Demographics Score) Analysis")
        print("="*60)
        
        # Check if BDS results already exist
        from config.paths import PATHS
        bds_file = PATHS.get_analysis_file_paths()['bds_table']
        if bds_file.exists():
            print(f"✅ BDS results already exist: {bds_file}")
            print("   Skipping BDS analysis step...")
            self.results['bds'] = pd.read_csv(bds_file)
            print(f"📄 BDS results loaded: {len(self.results['bds'])} features")
            return True
        
        try:
            success = self.bds_analyzer.run_analysis()
            if success:
                print("✅ BDS analysis completed successfully")
                # Load BDS results
                if bds_file.exists():
                    self.results['bds'] = pd.read_csv(bds_file)
                    print(f"📄 BDS results loaded: {len(self.results['bds'])} features")
                else:
                    print("⚠️ BDS results file not found")
            else:
                print("❌ BDS analysis failed")
                return False
        except Exception as e:
            print(f"❌ BDS analysis error: {e}")
            return False
        
        return True
    
    def run_vac_analysis(self):
        """Run Visual Attribute Consistency Score analysis"""
        print("\n" + "="*60)
        print("📊 2. Running VAC (Visual Attribute Consistency Score) Analysis")
        print("="*60)
        
        # Check if VAC results already exist
        from config.paths import PATHS
        analysis_paths = PATHS.get_analysis_file_paths()
        vac_matrix_file = analysis_paths['vac_matrix']
        if vac_matrix_file.exists():
            print(f"✅ VAC results already exist: {vac_matrix_file}")
            print("   Skipping VAC analysis step...")
            self.results['vac'] = {}
            vac_files = {
                'matrix': vac_matrix_file,
                'perfect_segregation': analysis_paths['vac_perfect_segregation'],
                'dramatic_shifts': analysis_paths['vac_dramatic_shifts']
            }
            
            for key, file_path in vac_files.items():
                if file_path.exists():
                    self.results['vac'][key] = pd.read_csv(file_path)
                    print(f"📄 VAC {key} loaded: {len(self.results['vac'][key])} rows")
            return True
        
        try:
            success = self.vac_analyzer.run_extraction()
            if success:
                print("✅ VAC analysis completed successfully")
                # Load VAC results
                vac_files = {
                    'matrix': analysis_paths['vac_matrix'],
                    'perfect_segregation': analysis_paths['vac_perfect_segregation'],
                    'dramatic_shifts': analysis_paths['vac_dramatic_shifts']
                }
                
                self.results['vac'] = {}
                for key, file_path in vac_files.items():
                    if file_path.exists():
                        self.results['vac'][key] = pd.read_csv(file_path)
                        print(f"📄 VAC {key} loaded: {len(self.results['vac'][key])} rows")
                    else:
                        print(f"⚠️ VAC {key} file not found")
            else:
                print("❌ VAC analysis failed")
                return False
        except Exception as e:
            print(f"❌ VAC analysis error: {e}")
            return False
        
        return True
    
    def run_cds_analysis(self):
        """Run Cross-Demographic Diversity Score analysis"""
        print("\n" + "="*60)
        print("📊 3. Running CDS (Cross-Demographic Diversity Score) Analysis")
        print("="*60)
        
        # Check if CDS results already exist
        from config.paths import PATHS
        analysis_paths = PATHS.get_analysis_file_paths()
        cds_file = analysis_paths['cds_matrix']
        if cds_file.exists():
            print(f"✅ CDS results already exist: {cds_file}")
            print("   Skipping CDS analysis step...")
            self.results['cds'] = pd.read_csv(cds_file)
            print(f"📄 CDS results loaded: {len(self.results['cds'])} features")
            return True
        
        try:
            success = self.cds_analyzer.run_analysis()
            if success:
                print("✅ CDS analysis completed successfully")
                # Load CDS results
                if cds_file.exists():
                    self.results['cds'] = pd.read_csv(cds_file)
                    print(f"📄 CDS results loaded: {len(self.results['cds'])} features")
                else:
                    print("⚠️ CDS results file not found")
            else:
                print("❌ CDS analysis failed")
                return False
        except Exception as e:
            print(f"❌ CDS analysis error: {e}")
            return False
        
        return True
    
    def create_comprehensive_summary(self):
        """Create comprehensive summary combining all three analyses"""
        print("\n" + "="*60)
        print("📊 4. Creating Comprehensive Summary")
        print("="*60)
        
        if not all([self.results['bds'] is not None, 
                   self.results['vac'] is not None, 
                   self.results['cds'] is not None]):
            print("❌ Cannot create summary - missing analysis results")
            return False
        
        try:
            # Create comprehensive feature summary
            summary_data = []
            
            # Get all unique features from all analyses
            bds_features = set(self.results['bds']['Feature'].unique()) if ('Feature' in self.results['bds'].columns and self.results['bds'] is not None) else set()
            vac_features = set(self.results['vac']['matrix'].columns[1:]) if ('matrix' in self.results['vac'] and self.results['vac']['matrix'] is not None) else set()
            cds_features = set(self.results['cds']['Feature'].unique()) if ('Feature' in self.results['cds'].columns and self.results['cds'] is not None) else set()
            
            all_features = bds_features | vac_features | cds_features
            
            print(f"📊 Total unique features across all analyses: {len(all_features)}")
            print(f"   - BDS features: {len(bds_features)}")
            print(f"   - VAC features: {len(vac_features)}")
            print(f"   - CDS features: {len(cds_features)}")
            
            for feature in sorted(all_features):
                # BDS scores
                bds_data = self.results['bds'][self.results['bds']['Feature'] == feature] if ('Feature' in self.results['bds'].columns and self.results['bds'] is not None) else pd.DataFrame()
                bds_score = bds_data['JS_Divergence'].mean() if (not bds_data.empty and 'JS_Divergence' in bds_data.columns) else 0.0
                
                # VAC scores
                vac_score = 0.0
                if ('matrix' in self.results['vac'] and self.results['vac']['matrix'] is not None and 
                    feature in self.results['vac']['matrix'].columns):
                    vac_score = self.results['vac']['matrix'][feature].mean()
                
                # CDS scores
                cds_data = self.results['cds'][self.results['cds']['Feature'] == feature] if ('Feature' in self.results['cds'].columns and self.results['cds'] is not None) else pd.DataFrame()
                cds_age = cds_data['Age Diversity'].mean() if (not cds_data.empty and 'Age Diversity' in cds_data.columns) else 0.0
                cds_gender = cds_data['Gender Diversity'].mean() if (not cds_data.empty and 'Gender Diversity' in cds_data.columns) else 0.0
                cds_ethnicity = cds_data['Ethnicity Diversity'].mean() if (not cds_data.empty and 'Ethnicity Diversity' in cds_data.columns) else 0.0
                cds_overall = (cds_age + cds_gender + cds_ethnicity) / 3
                
                # Calculate composite bias score
                composite_score = (bds_score + vac_score + cds_overall) / 3
                
                # Determine bias level
                if composite_score >= 0.6:
                    bias_level = "High"
                elif composite_score >= 0.3:
                    bias_level = "Medium"
                else:
                    bias_level = "Low"
                
                summary_data.append({
                    'Feature': feature,
                    'BDS_Score': round(bds_score, 3),
                    'VAC_Score': round(vac_score, 3),
                    'CDS_Age': round(cds_age, 3),
                    'CDS_Gender': round(cds_gender, 3),
                    'CDS_Ethnicity': round(cds_ethnicity, 3),
                    'CDS_Overall': round(cds_overall, 3),
                    'Composite_Score': round(composite_score, 3),
                    'Bias_Level': bias_level
                })
            
            # Create summary DataFrame
            self.results['summary'] = pd.DataFrame(summary_data)
            self.results['summary'] = self.results['summary'].sort_values('Composite_Score', ascending=False)
            
            print(f"✅ Comprehensive summary created with {len(self.results['summary'])} features")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating comprehensive summary: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self):
        """Save all analysis results"""
        print("\n" + "="*60)
        print("💾 5. Saving Analysis Results")
        print("="*60)
        
        try:
            # Save comprehensive summary - DISABLED
            # if self.results['summary'] is not None:
            #     summary_file = self.analysis_output_dir / "comprehensive_bias_summary.csv"
            #     self.results['summary'].to_csv(summary_file, index=False)
            #     print(f"📄 Comprehensive summary saved: {summary_file}")
            print("📄 Comprehensive summary generation disabled")
            
            # Copy individual analysis results to comprehensive folder

            
            # Create analysis metadata

            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def print_summary_statistics(self):
        """Print summary statistics"""
        if self.results['summary'] is None:
            print("📊 Comprehensive summary not available (summary generation disabled)")
            print("   Individual analysis results are still available in the output directory")
            return
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE BIAS ANALYSIS SUMMARY")
        print("="*60)
        
        summary_df = self.results['summary']
        
        print(f"🎯 Total Features Analyzed: {len(summary_df)}")
        print(f"\n📈 Bias Level Distribution:")
        bias_counts = summary_df['Bias_Level'].value_counts()
        for level in ['High', 'Medium', 'Low']:
            count = bias_counts.get(level, 0)
            percentage = (count / len(summary_df)) * 100
            print(f"   {level}: {count} features ({percentage:.1f}%)")
        
        print(f"\n🏆 Top 10 Most Biased Features:")
        print("="*60)
        top_features = summary_df.head(10)[['Feature', 'BDS_Score', 'VAC_Score', 'CDS_Overall', 'Composite_Score', 'Bias_Level']]
        print(top_features.to_string(index=False))
        
        print(f"\n📊 Average Scores by Analysis Method:")
        print(f"   BDS Average: {summary_df['BDS_Score'].mean():.3f}")
        print(f"   VAC Average: {summary_df['VAC_Score'].mean():.3f}")
        print(f"   CDS Average: {summary_df['CDS_Overall'].mean():.3f}")
        print(f"   Composite Average: {summary_df['Composite_Score'].mean():.3f}")
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive bias analysis"""
        print("🚀 Starting SODA Comprehensive Bias Analysis")
        print("="*80)
        
        # Check if comprehensive analysis results already exist
        summary_file = self.analysis_output_dir / "comprehensive_bias_summary.csv"
        if summary_file.exists():
            print(f"✅ Comprehensive analysis results already exist: {summary_file}")
            print("   Skipping comprehensive analysis step...")
            return True
        
        # Run all three analyses
        analyses = [
            ("BDS", self.run_bds_analysis),
            ("VAC", self.run_vac_analysis),
            ("CDS", self.run_cds_analysis)
        ]
        
        for name, analysis_func in analyses:
            success = analysis_func()
            if not success:
                print(f"❌ {name} analysis failed - stopping comprehensive analysis")
                return False
        
        # Create comprehensive summary - DISABLED
        # if not self.create_comprehensive_summary():
        #     print("❌ Failed to create comprehensive summary")
        #     return False
        print("📊 Comprehensive summary creation disabled")
        
        # Save results
        if not self.save_results():
            print("❌ Failed to save results")
            return False
        
        # Print summary
        self.print_summary_statistics()
        
        print("\n" + "="*80)
        print("✅ SODA Comprehensive Bias Analysis Completed Successfully!")
        print(f"📁 Results saved to: {self.analysis_output_dir}")
        print("="*80)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='SODA Comprehensive Bias Analysis Framework')
    parser.add_argument('--output_dir', default='comprehensive_analysis', 
                       help='Output directory for analysis results (default: comprehensive_analysis)')
    parser.add_argument('--data_dir', default='/home/allsound/SODA/outputs',
                       help='Input data directory (default: /home/allsound/SODA/outputs)')
    
    args = parser.parse_args()
    
    # Run comprehensive analysis
    analyzer = ComprehensiveBiasAnalyzer(
        output_base_dir=args.data_dir,
        analysis_output_dir=args.output_dir
    )
    
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n💡 Next Steps:")
        print("   1. Review comprehensive_bias_summary.csv for overall results")
        print("   2. Examine individual analysis files for detailed insights")

        sys.exit(0)
    else:
        print("\n❌ Comprehensive analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()