#!/usr/bin/env python3
"""
SODA Comprehensive Bias Analysis Runner

This script runs the comprehensive bias analysis combining BDS, VAC, and CDS methods.
Can be run independently after generating images and basic analysis.

Usage:
    python run_comprehensive_analysis.py [options]
    
Examples:
    # Run with default settings
    python run_comprehensive_analysis.py
    
    # Specify custom directories
    python run_comprehensive_analysis.py --data_dir ./outputs --output_dir ./final_analysis
"""

import sys
import os
from pathlib import Path

# Add the SODA directory to Python path for imports
soda_dir = Path(__file__).parent
sys.path.insert(0, str(soda_dir))

from analysis.comprehensive_bias_analyzer import ComprehensiveBiasAnalyzer

def main():
    """Run comprehensive bias analysis"""
    
    print("🚀 SODA Comprehensive Bias Analysis Runner")
    print("="*80)
    print("This tool integrates three bias analysis methods:")
    print("  📊 BDS: Baseline vs Demographics Score")
    print("  📊 VAC: Visual Attribute Consistency Score") 
    print("  📊 CDS: Cross-Demographic Diversity Score")
    print("="*80)
    
    # Check if outputs directory exists
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ Error: 'outputs' directory not found!")
        print("   Please run the main SODA framework first to generate data.")
        print("   Example: python main_framework.py --model gpt --objects car laptop")
        return False
    
    # Check if we have generated data
    output_folders = list(outputs_dir.glob("*_*"))
    if not output_folders:
        print("❌ Error: No model-object output folders found in 'outputs' directory!")
        print("   Please run the main SODA framework first to generate data.")
        return False
    
    print(f"📁 Found {len(output_folders)} model-object combinations:")
    for folder in sorted(output_folders):
        print(f"   - {folder.name}")
    
    # Run comprehensive analysis
    try:
        analyzer = ComprehensiveBiasAnalyzer(
            output_base_dir=str(outputs_dir),
            analysis_output_dir="outputs/comprehensive_analysis"
        )
        
        success = analyzer.run_comprehensive_analysis()
        
        if success:
            print("\n" + "="*80)
            print("✅ SODA Comprehensive Bias Analysis completed successfully!")
            print("="*80)
            print("\n📁 Results saved to: comprehensive_analysis/")
            print("\n📄 Key output files:")
            print("   - comprehensive_bias_summary.csv    (Main results)")
            print("   - bds_score/baseline_vs_8_demographics_table.csv (Baseline vs Demographics)")
            print("   - vac_scores/vac_matrix.csv         (Visual Attribute Consistency)")
            print("   - cds_score/cds_feature_diversity_matrix.csv (Cross-Demographic Diversity)")
            
            print("\n💡 Next steps:")
            print("   1. Review comprehensive_bias_summary.csv for overall bias scores")
            print("   2. Examine individual analysis files for detailed insights")
            print("   3. Use the results for your research paper or report")
            
            return True
        else:
            print("\n❌ Comprehensive analysis failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)