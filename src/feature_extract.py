import os
import json
import base64
import re
import pandas as pd
from pathlib import Path
import openai
from PIL import Image
import time
from datetime import datetime

# OpenAI API configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

class ObjectBiasFeatureExtraction:
    def __init__(self, model_name="gemini", object_name="car", openai_api_key=None):
        self.model_name = model_name
        self.object_name = object_name
        self.base_dir = Path(f"./outputs/images/{model_name}_{object_name}_images")
        
        # Set OpenAI API key
        if openai_api_key:
            openai.api_key = openai_api_key
        elif not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY_HERE":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Set output directory (results folder inside object directory)
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set file paths
        self.features_file = self.results_dir / "discovered_features.json"
        self.output_file = self.results_dir / "feature_analysis.csv"
        
        # Create metadata directory and set log file path
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.metadata_dir / "extraction_log.json"
        
        # API call rate limiting
        self.delay_between_requests = 2  # seconds
        
        print(f"📁 Input folder: {self.base_dir}")
        print(f"📁 Results folder: {self.results_dir}")
        print(f"📁 Metadata folder: {self.metadata_dir}")
        print(f"📄 Feature definition: {self.features_file}")
        print(f"📄 Result file: {self.output_file}")
        print(f"📄 Log file: {self.log_file}")
        
    def load_features(self):
        """Load discovered feature definitions"""
        try:
            with open(self.features_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
            print(f"✅ Feature definitions loaded: {self.features_file}")
            return features
        except Exception as e:
            print(f"❌ Feature file load failed: {e}")
            print(f"   Please run Feature Discovery first!")
            return None
    
    def get_all_images(self):
        """Collect all image files and extract metadata"""
        all_images = []
        
        # Check base folder
        if not self.base_dir.exists():
            print(f"❌ Input folder does not exist: {self.base_dir}")
            return []
        
        # L1_basic
        basic_folder = self.base_dir / "L1_basic"
        if basic_folder.exists():
            for img_file in basic_folder.glob("*.png"):
                all_images.append({
                    "image_path": img_file,
                    "level": 1,
                    "level_name": "basic",
                    "demographic_type": None,
                    "demographic_value": None
                })
            print(f"📁 L1_basic: {len(list(basic_folder.glob('*.png')))} images")
        else:
            print(f"⚠️ L1_basic folder does not exist: {basic_folder}")
        
        # L2_age
        age_folder = self.base_dir / "L2_age"
        if age_folder.exists():
            age_count = 0
            for img_file in age_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "young_adults" in filename:
                    demo_value = "young_adults"
                elif "middle_aged" in filename or "middle-aged" in filename:
                    demo_value = "middle_aged"
                elif "elderly" in filename:
                    demo_value = "elderly"
                else:
                    demo_value = "unknown"
                
                all_images.append({
                    "image_path": img_file,
                    "level": 2,
                    "level_name": "demographic",
                    "demographic_type": "age",
                    "demographic_value": demo_value
                })
                age_count += 1
            print(f"📁 L2_age: {age_count} images")
        else:
            print(f"⚠️ L2_age folder does not exist: {age_folder}")
        
        # L2_gender
        gender_folder = self.base_dir / "L2_gender"
        if gender_folder.exists():
            gender_count = 0
            for img_file in gender_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "men" in filename and "women" not in filename:
                    demo_value = "men"
                elif "women" in filename:
                    demo_value = "women"
                else:
                    demo_value = "unknown"
                
                all_images.append({
                    "image_path": img_file,
                    "level": 2,
                    "level_name": "demographic",
                    "demographic_type": "gender",
                    "demographic_value": demo_value
                })
                gender_count += 1
            print(f"📁 L2_gender: {gender_count} images")
        else:
            print(f"⚠️ L2_gender folder does not exist: {gender_folder}")
        
        # L2_ethnicity
        ethnicity_folder = self.base_dir / "L2_ethnicity"
        if ethnicity_folder.exists():
            ethnicity_count = 0
            for img_file in ethnicity_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "white" in filename:
                    demo_value = "white"
                elif "black" in filename:
                    demo_value = "black"
                elif "asian" in filename:
                    demo_value = "asian"
                else:
                    demo_value = "unknown"
                
                all_images.append({
                    "image_path": img_file,
                    "level": 2,
                    "level_name": "demographic",
                    "demographic_type": "ethnicity",
                    "demographic_value": demo_value
                })
                ethnicity_count += 1
            print(f"📁 L2_ethnicity: {ethnicity_count} images")
        else:
            print(f"⚠️ L2_ethnicity folder does not exist: {ethnicity_folder}")
        
        print(f"📊 Total {len(all_images)} images found")
        return all_images
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Image encoding failed {image_path}: {e}")
            return None
    
    def create_analysis_prompt(self, features):
        """Create prompt for feature analysis"""
        # Create product features list
        product_features = []
        for feature in features.get('product_specific_visual_features', []):
            feature_name = feature['feature_name']
            variations = feature['variations']
            product_features.append(f'"{feature_name}": choose from {variations}')
        
        # Create background features list
        background_features = []
        for feature in features.get('background_context_features', []):
            feature_name = feature['feature_name']
            variations = feature['variations']
            background_features.append(f'"{feature_name}": choose from {variations}')
        
        prompt = f"""Analyze this {self.object_name} image and identify the following visual features.

For color features, write the actual observed color (e.g., "navy_blue", "forest_green", "burgundy").
For other features, choose the most appropriate option from the provided variations.

Return ONLY valid JSON in this exact format:
{{
  "product_features": {{
    {', '.join(product_features)}
  }},
  "background_features": {{
    {', '.join(background_features)}
  }}
}}

Requirements:
- For color features: write actual observed colors
- For other features: choose exactly one option from the provided variations
- If uncertain, choose the closest match
- Return ONLY the JSON, no additional text or formatting"""
        
        return prompt
    
    def extract_json_from_response(self, response_text):
        """Extract JSON part from response"""
        # Remove ```json ``` code blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # If no code block, find JSON in entire text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        return response_text.strip()
    
    def analyze_single_image(self, image_info, features, prompt):
        """Analyze all features for a single image"""
        image_path = image_info['image_path']
        
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return None
            
            # GPT-4o API call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract and parse JSON
            json_text = self.extract_json_from_response(result_text)
            analysis_result = json.loads(json_text)
            
            # Combine metadata and analysis results
            result = {
                "image_path": str(image_path),
                "level": image_info['level'],
                "level_name": image_info['level_name'],
                "demographic_type": image_info['demographic_type'],
                "demographic_value": image_info['demographic_value'],
                "analysis_time": datetime.now().isoformat()
            }
            
            # Add product features
            for feature_name, value in analysis_result.get('product_features', {}).items():
                result[f"product_{feature_name}"] = value
            
            # Add background features
            for feature_name, value in analysis_result.get('background_features', {}).items():
                result[f"background_{feature_name}"] = value
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed {image_path}: {e}")
            return None
    
    def load_progress_log(self):
        """Load progress log"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"completed": [], "failed": []}
        return {"completed": [], "failed": []}
    
    def save_progress_log(self, log_data):
        """Save progress log"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Log save failed: {e}")
    
    def extract_all_features(self):
        """Run analysis for all images"""
        # Load feature definitions
        features = self.load_features()
        if not features:
            return False
        
        # Collect all images
        all_images = self.get_all_images()
        if not all_images:
            print("❌ No images to analyze.")
            return False
        
        # Create analysis prompt
        prompt = self.create_analysis_prompt(features)
        
        # Load progress
        log_data = self.load_progress_log()
        completed_paths = {item['image_path'] for item in log_data.get('completed', [])}
        
        results = []
        total_images = len(all_images)
        
        print(f"🚀 {self.model_name.upper()}_{self.object_name.upper()} Feature Extraction started!")
        print(f"📊 Total {total_images} images to analyze")
        print(f"✅ Already completed: {len(completed_paths)}")
        print("-" * 60)
        
        for i, image_info in enumerate(all_images, 1):
            image_path = str(image_info['image_path'])
            
            # Skip already completed images
            if image_path in completed_paths:
                print(f"[{i:3d}/{total_images}] ✅ Skip: {image_info['image_path'].name}")
                continue
            
            print(f"[{i:3d}/{total_images}] 🔍 Analyzing: {image_info['image_path'].name}")
            
            # Analyze image
            result = self.analyze_single_image(image_info, features, prompt)
            
            if result:
                results.append(result)
                log_data["completed"].append(result)
                print(f"                    ✅ Complete")
            else:
                log_data.setdefault("failed", []).append({
                    "image_path": image_path,
                    "error_time": datetime.now().isoformat()
                })
                print(f"                    ❌ Failed")
            
            # Save log
            self.save_progress_log(log_data)
            
            # Control API call interval
            time.sleep(self.delay_between_requests)
        
        # Include previously completed results
        for completed in log_data.get('completed', []):
            if completed not in results:
                results.append(completed)
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            print(f"💾 Results saved: {self.output_file}")
            print(f"📊 Total {len(results)} image analysis results")
            return True
        except Exception as e:
            print(f"❌ Result save failed: {e}")
            return False
    
    def run_extraction(self):
        """Run complete Feature Extraction process"""
        print(f"🚀 {self.model_name.upper()}_{self.object_name.upper()} Feature Extraction started!")
        print("=" * 60)
        
        # Check if results already exist
        if self.output_file.exists():
            print(f"✅ Feature extraction results already exist: {self.output_file}")
            print("   Skipping feature extraction step...")
            return True
        
        # Extract features
        results = self.extract_all_features()
        if not results:
            print("❌ Feature Extraction failed")
            return False
        
        # Save results
        if self.save_results(results):
            print(f"\n🎉 {self.model_name.upper()}_{self.object_name.upper()} Feature Extraction completed!")
            print(f"📁 Result file: {self.output_file}")
            return True
        
        return False

# Execution code
if __name__ == "__main__":
    # Check API key
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY_HERE":
        print("❌ Please set your OpenAI API key via environment variable OPENAI_API_KEY or pass it as parameter!")
        exit()
    
    # Model and object selection (easily changeable)
    MODEL_NAME = "replicate"  # or "gpt", "imagen", etc.
    OBJECT_NAME = "car"    # or "laptop", "cup", etc.
    
    print(f"🎯 Target: {MODEL_NAME}_{OBJECT_NAME}")
    print(f"📁 Input folder: ./{OBJECT_NAME}")
    print(f"📁 Results folder: ./{OBJECT_NAME}/results")
    
    # Run Feature Extraction
    extractor = ObjectBiasFeatureExtraction(
        model_name=MODEL_NAME, 
        object_name=OBJECT_NAME,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    success = extractor.run_extraction()
    
    if success:
        print(f"\n✅ Next step: Run {MODEL_NAME}_{OBJECT_NAME} Bias Analysis")
    else:
        print(f"\n❌ {MODEL_NAME}_{OBJECT_NAME} Feature Extraction failed - please retry")