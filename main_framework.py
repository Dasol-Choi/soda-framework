#!/usr/bin/env python3
"""
SODA Main Framework 
Integrates prompt generation and image generation
"""

import argparse
import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import OUTPUT_BASE_DIR from image_generator
from src.image_generator import OUTPUT_BASE_DIR

def main():
    parser = argparse.ArgumentParser(description='SODA Framework')
    parser.add_argument('--model', required=True,
                       help='Model to use for image generation. For replicate, use format: replicate.model_name (e.g., replicate.sdxl, replicate.imagen4)')
    parser.add_argument('--objects', nargs='+', required=True,
                       help='Objects to analyze')
    parser.add_argument('--prompt-only', action='store_true',
                       help='Generate prompts only, skip image generation')
    parser.add_argument('--image-only', action='store_true',
                       help='Generate images only, skip prompt generation (requires existing prompts)')
    parser.add_argument('--images-per-prompt', type=int, default=2,
                       help='Number of images to generate per prompt (default: 2)')
    
    args = parser.parse_args()
    
    # Parse model argument for replicate sub-models
    model_name = args.model
    replicate_submodel = None
    
    if args.model.startswith('replicate.'):
        model_parts = args.model.split('.')
        if len(model_parts) == 2:
            model_name = 'replicate'
            replicate_submodel = model_parts[1]
        else:
            print("❌ Invalid replicate model format. Use: replicate.model_name (e.g., replicate.sdxl)")
            return None
    
    print("=== SODA Framework ===")
    print(f"Model: {args.model}")
    if replicate_submodel:
        print(f"Replicate sub-model: {replicate_submodel}")
    print(f"Objects: {args.objects}")
    print(f"Images per prompt: {args.images_per_prompt}")
    print(f"Prompt only: {args.prompt_only}")
    print(f"Image only: {args.image_only}")
    
    # Step 1: Prompt Generation
    if not args.image_only:
        print("\n=== Step 1: Prompt Generation ===")
        from src.prompt_generator import ObjectBiasPromptGenerator
        
        prompt_generator = ObjectBiasPromptGenerator(args.objects)
        prompt_result = prompt_generator.run_for_objects(args.objects)
        
        print(f"✓ Generated {len(prompt_result['prompts'])} prompts")
        
        if args.prompt_only:
            print("✓ Prompt generation complete. Exiting.")
            return prompt_result
    
    # Step 2: Image Generation
    if not args.prompt_only:
        print("\n=== Step 2: Image Generation ===")
        from src.image_generator import UnifiedImageGenerator
        
        # Initialize only the needed generator based on model
        if model_name == 'gpt':
            from src.image_generator import GPTImageGenerator
            generator = GPTImageGenerator()
        elif model_name == 'replicate':
            # Replicate generator will be initialized per object
            generator = None
        elif model_name == 'imagen':
            # Imagen generator will be initialized per object
            generator = None
        else:
            # For other models, initialize per object
            generator = None
        
        if args.image_only:
            # Load existing prompts from CSV
            import pandas as pd
            from config.paths import PATHS
            prompts = []
            for obj in args.objects:
                csv_path = PATHS.get_prompt_file_path(obj)
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    prompts.extend(df.to_dict('records'))
                else:
                    print(f"⚠️ Warning: {csv_path} not found")
            
            if not prompts:
                print("❌ No prompts found. Run prompt generation first.")
                return None
        else:
            # Use prompts from step 1
            prompts = prompt_result['prompts']
        
        # Set global images per prompt
        from src.image_generator import IMAGES_PER_PROMPT
        import src.image_generator as image_generator_module
        image_generator_module.IMAGES_PER_PROMPT = args.images_per_prompt
        
        # Generate images for each object
        all_generated_files = []
        for obj in args.objects:
            print(f"\n--- Generating images for {obj} ---")
            
            # Create generator using factory pattern
            try:
                from src.model_factory import ModelFactory
                
                generator_kwargs = {}
                if model_name == 'replicate' and replicate_submodel:
                    generator_kwargs['replicate_submodel'] = replicate_submodel
                
                generator = ModelFactory.create_generator(
                    model_name=model_name,
                    object_name=obj,
                    **generator_kwargs
                )
                
                # Generate all images for this object
                generated_files = generator.generate_all_images()
                all_generated_files.extend(generated_files)
                
                # Log file is now automatically saved to metadata folder
                print(f"📋 Generation log saved to metadata folder")
                
            except ValueError as e:
                print(f"❌ Error creating generator: {e}")
                continue
        
        print(f"✓ Generated {len([f for f in all_generated_files if f is not None])} images")
    
    # Step 3: Analysis
    if not args.prompt_only and not args.image_only:
        print("\n=== Step 3: Analysis ===")
        
        # Import analysis modules
        from src.feature_gen import ObjectBiasFeatureDiscovery
        from src.feature_extract import ObjectBiasFeatureExtraction
        from analysis.analysis_bias import ObjectBiasAnalyzer
        from analysis.analysis_js import ObjectBiasJSDivergenceAnalyzer
    
        
        for obj in args.objects:
            print(f"\n--- Analyzing {obj} ---")
            
            # Determine model name for analysis
            if model_name == 'replicate' and replicate_submodel:
                # Use the same folder mapping logic for analysis
                folder_mapping = {
                    'sdxl': 'diffusion',
                    'imagen4': 'imagen',
                    'flux-2-pro': 'flux',
                    'qwen-image': 'qwen'
                }
                analysis_model_name = folder_mapping.get(replicate_submodel, replicate_submodel)
            else:
                analysis_model_name = model_name
            
            # Step 3.1: Feature Discovery
            print(f"🔍 Step 3.1: Feature Discovery for {analysis_model_name}_{obj}")
            feature_discovery = ObjectBiasFeatureDiscovery(
                model_name=analysis_model_name,
                object_name=obj
            )
            discovery_success = feature_discovery.run_discovery()
            
            if discovery_success:
                # Step 3.2: Feature Extraction
                print(f"📊 Step 3.2: Feature Extraction for {analysis_model_name}_{obj}")
                feature_extraction = ObjectBiasFeatureExtraction(
                    model_name=analysis_model_name,
                    object_name=obj
                )
                extraction_success = feature_extraction.run_extraction()
                
                if extraction_success:
                    # Step 3.3: Bias Analysis
                    print(f"⚖️ Step 3.3: Bias Analysis for {analysis_model_name}_{obj}")
                    bias_analyzer = ObjectBiasAnalyzer(
                        model_name=analysis_model_name,
                        object_name=obj
                    )
                    bias_success = bias_analyzer.run_comprehensive_analysis()
                    
                    if bias_success:
                        # Step 3.4: JS Divergence Analysis
                        print(f"📈 Step 3.4: JS Divergence Analysis for {analysis_model_name}_{obj}")
                        js_analyzer = ObjectBiasJSDivergenceAnalyzer(
                            model_name=analysis_model_name,
                            object_name=obj
                        )
                        js_success = js_analyzer.run_divergence_analysis()
                        
                        if js_success:
                            print(f"✅ {analysis_model_name}_{obj} individual analysis completed successfully!")
                        else:
                            print(f"❌ {analysis_model_name}_{obj} JS Divergence analysis failed")
                    else:
                        print(f"❌ {analysis_model_name}_{obj} bias analysis failed")
                else:
                    print(f"❌ {analysis_model_name}_{obj} feature extraction failed")
            else:
                print(f"❌ {analysis_model_name}_{obj} feature discovery failed")
    
    # Step 4: Comprehensive Bias Analysis (integrates BDS, VAC, CDS)
    print("\n" + "="*60)
    print("📊 Step 4: SODA Comprehensive Bias Analysis")
    print("="*60)
    
    try:
        from analysis.comprehensive_bias_analyzer import ComprehensiveBiasAnalyzer
        
        comprehensive_analyzer = ComprehensiveBiasAnalyzer(
            output_base_dir="outputs",
            analysis_output_dir="outputs/comprehensive_analysis"
        )
        
        comprehensive_success = comprehensive_analyzer.run_comprehensive_analysis()
        
        if comprehensive_success:
            print("✅ SODA Comprehensive Bias Analysis completed successfully!")
        else:
            print("❌ SODA Comprehensive Bias Analysis failed")
    
    except ImportError as e:
        print(f"⚠️ Comprehensive analysis not available: {e}")
        print("   Please ensure analysis modules are in the analysis/ directory")
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {e}")
    
    print("\n=== SODA Framework Complete ===")
    return {
        'model': args.model,
        'objects': args.objects,
        'prompts': prompts if 'prompts' in locals() else None,
        'generated_files': all_generated_files if 'all_generated_files' in locals() else None
    }

if __name__ == "__main__":
    result = main() 