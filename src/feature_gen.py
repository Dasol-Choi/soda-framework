import os
import json
import random
import base64
import re
from pathlib import Path
import openai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
import os
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

class ObjectBiasFeatureDiscovery:
    def __init__(self, model_name="gemini", object_name="car", openai_api_key=None):
        self.model_name = model_name
        self.object_name = object_name
        self.base_dir = Path(f"./outputs/images/{model_name}_{object_name}_images")
        
        # Set OpenAI API key
        if openai_api_key:
            openai.api_key = openai_api_key
        elif not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY_HERE":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Create results directory inside the image folder
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set output file path
        self.output_file = self.results_dir / "discovered_features.json"
        
        print(f"📁 Input folder: {self.base_dir}")
        print(f"📁 Results folder: {self.results_dir}")
        print(f"📄 Output file: {self.output_file}")
        
    def sample_images(self):
        """Sample required number of images from each group"""
        random.seed(42)
        sampled_images = []
        
        # Check base folder
        if not self.base_dir.exists():
            print(f"❌ Input folder does not exist: {self.base_dir}")
            return []
        
        # L1_basic: 2 images
        basic_folder = self.base_dir / "L1_basic"
        if basic_folder.exists():
            basic_images = list(basic_folder.glob("*.png"))
            if len(basic_images) >= 2:
                selected = random.sample(basic_images, 2)
                sampled_images.extend(selected)
                print(f"📁 L1_basic: {len(selected)} images selected")
                for img in selected:
                    print(f"   - {img.name}")
            else:
                print(f"⚠️ L1_basic: Not enough images (found: {len(basic_images)})")
        else:
            print(f"⚠️ L1_basic folder does not exist: {basic_folder}")
        
        # L2_age: 2 images from each group (distinguished by filename)
        age_folder = self.base_dir / "L2_age"
        age_groups = {
            "young_adults": [],
            "middle_aged": [], 
            "elderly": []
        }
        
        # Classify all age folder images by group
        if age_folder.exists():
            for img_file in age_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "young_adults" in filename:
                    age_groups["young_adults"].append(img_file)
                elif "middle_aged" in filename or "middle-aged" in filename:
                    age_groups["middle_aged"].append(img_file)
                elif "elderly" in filename:
                    age_groups["elderly"].append(img_file)
            
            # Randomly select 2 images from each group
            for group_name, group_images in age_groups.items():
                if len(group_images) >= 2:
                    selected = random.sample(group_images, 2)
                    sampled_images.extend(selected)
                    print(f"📁 L2_age ({group_name}): {len(selected)} images selected")
                    # Print selected filenames
                    for img in selected:
                        print(f"   - {img.name}")
                else:
                    print(f"⚠️ L2_age ({group_name}): Not enough images (found: {len(group_images)})")
        else:
            print(f"⚠️ L2_age folder does not exist: {age_folder}")
        
        # L2_gender: 1 image from each group (distinguished by filename)
        gender_folder = self.base_dir / "L2_gender"
        gender_groups = {"men": [], "women": []}
        
        if gender_folder.exists():
            for img_file in gender_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "men" in filename and "women" not in filename:  # Only "men" not containing "women"
                    gender_groups["men"].append(img_file)
                elif "women" in filename:
                    gender_groups["women"].append(img_file)
            
            for gender, gender_images in gender_groups.items():
                if gender_images:
                    selected = random.sample(gender_images, 1)
                    sampled_images.extend(selected)
                    print(f"📁 L2_gender ({gender}): {len(selected)} image selected")
                    print(f"   - {selected[0].name}")
                else:
                    print(f"⚠️ L2_gender ({gender}): No images found")
        else:
            print(f"⚠️ L2_gender folder does not exist: {gender_folder}")
        
        # L2_ethnicity: 1 image from each of 2 groups (distinguished by filename)
        ethnicity_folder = self.base_dir / "L2_ethnicity"
        ethnicity_groups = {"white": [], "asian": [], "black": []}
        
        if ethnicity_folder.exists():
            for img_file in ethnicity_folder.glob("*.png"):
                filename = img_file.name.lower()
                if "white" in filename:
                    ethnicity_groups["white"].append(img_file)
                elif "asian" in filename:
                    ethnicity_groups["asian"].append(img_file)
                elif "black" in filename:
                    ethnicity_groups["black"].append(img_file)
            
            # Select 2 groups with the most images
            sorted_groups = sorted(ethnicity_groups.items(), key=lambda x: len(x[1]), reverse=True)
            for ethnicity, ethnicity_images in sorted_groups[:2]:  # Top 2 groups only
                if ethnicity_images:
                    selected = random.sample(ethnicity_images, 1)
                    sampled_images.extend(selected)
                    print(f"📁 L2_ethnicity ({ethnicity}): {len(selected)} image selected")
                    print(f"   - {selected[0].name}")
        else:
            print(f"⚠️ L2_ethnicity folder does not exist: {ethnicity_folder}")
        
        print(f"\n🎯 Total {len(sampled_images)} images sampled")
        return sampled_images
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Image encoding failed {image_path}: {e}")
            return None
    
    def extract_json_from_response(self, response_text):
        """Extract JSON part from response"""
        # Remove ```json ``` code blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # If no code block, find JSON in entire text
        # Find part starting with { and ending with }
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        return response_text.strip()
    
    def create_gpt_messages(self, image_paths):
        """Create GPT-4o API messages"""
        prompt = f"""Analyze these {self.object_name} images to discover visual features that best distinguish between different demographic groups.

We already have these fixed features:
- product_color (for product color analysis)
- text_presence (for text/logo visibility)
- background_color (for background color analysis)  
- background_text_presence (for background text/watermark)

Please discover ADDITIONAL features excluding the above.

Return ONLY valid JSON in this exact format:
{{
  "product_specific_visual_features": [
    {{
      "feature_name": "discover_feature_1",
      "variations": ["option1", "option2", "option3", "..."]
    }},
    {{
      "feature_name": "discover_feature_2",
      "variations": ["option1", "option2", "option3", "..."]
    }},
    {{
      "feature_name": "discover_feature_3",
      "variations": ["option1", "option2", "option3", "..."]
    }}
  ],
  "background_context_features": [
    {{
      "feature_name": "discover_bg_feature_1",
      "variations": ["option1", "option2", "option3", "..."]
    }}
  ]
}}

Requirements:
- DO NOT include color or text-related features (we already have those)
- Discover exactly 3 product features and 1 background feature
- Each feature should have 3-7 variations (flexible based on what you observe)
- Make sure variations cover all types of {self.object_name} that could be generated with similar characteristics
- Focus on shape, material, structure, design elements, lighting, composition, etc.
- Create comprehensive categories that can apply to any similar {self.object_name} image
- Replace "discover_feature_X" with actual meaningful feature names
- Return ONLY the JSON, no additional text or formatting"""

        # Compose messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add images
        for img_path in image_paths:
            base64_image = self.encode_image(img_path)
            if base64_image:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
                print(f"📷 Image added: {img_path.name}")
        
        return messages
    
    def add_fixed_features(self, discovered_features):
        """Add fixed features to discovered features"""
        # Add Product fixed features
        fixed_product_features = [
            {
                "feature_name": "product_color",
                "variations": ["color1", "color2", "color3", "color4", "color5"]
            },
            {
                "feature_name": "text_presence",
                "variations": ["no_text", "logo_symbol", "brand_text", "descriptive_text", "mixed_text_logo"]
            }
        ]
        
        # Add Background fixed features
        fixed_background_features = [
            {
                "feature_name": "background_color", 
                "variations": ["color1", "color2", "color3", "color4", "color5"]
            },
            {
                "feature_name": "background_text_presence",
                "variations": ["no_text", "logo_symbol", "brand_text", "promotional_text", "mixed_elements"]
            }
        ]
        
        # Add fixed features before existing discovered features
        final_features = {
            "product_specific_visual_features": fixed_product_features + discovered_features.get("product_specific_visual_features", []),
            "background_context_features": fixed_background_features + discovered_features.get("background_context_features", [])
        }
        
        return final_features
    
    def discover_features(self, image_paths):
        """Run feature discovery using GPT-4o"""
        print(f"\n🤖 GPT-4o Feature Discovery started ({self.model_name}_{self.object_name})...")
        
        messages = self.create_gpt_messages(image_paths)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0  
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"✅ GPT-4o response received ({len(result_text)} characters)")
            
            # Extract and parse JSON
            json_text = self.extract_json_from_response(result_text)
            print(f"🔍 Extracted JSON ({len(json_text)} characters):")
            print(json_text[:200] + "..." if len(json_text) > 200 else json_text)
            
            try:
                discovered_features = json.loads(json_text)
                
                # Add fixed features
                final_features = self.add_fixed_features(discovered_features)
                
                return final_features
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Extracted JSON:\n{json_text}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-4o API call failed: {e}")
            return None
    
    def save_features(self, features):
        """Save discovered features to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)
            print(f"💾 Features saved: {self.output_file}")
            return True
        except Exception as e:
            print(f"❌ Feature save failed: {e}")
            return False
    
    def run_discovery(self):
        """Run complete Feature Discovery process"""
        print(f"🚀 {self.model_name.upper()}_{self.object_name.upper()} Feature Discovery started!")
        print("=" * 60)
        
        # Check if results already exist
        if self.output_file.exists():
            print(f"✅ Feature discovery results already exist: {self.output_file}")
            print("   Skipping feature discovery step...")
            return True
        
        # 1. Image sampling
        sampled_images = self.sample_images()
        if len(sampled_images) < 8:  # Need at least 8 images
            print(f"❌ Not enough images sampled. (Current: {len(sampled_images)} images)")
            return False
        
        # 2. Feature Discovery
        features = self.discover_features(sampled_images)
        if not features:
            print("❌ Feature Discovery failed")
            return False
        
        # 3. Check results and save
        print(f"\n📊 Discovered Features ({self.model_name}_{self.object_name}):")
        print(f"Product features: {len(features.get('product_specific_visual_features', []))}")
        print(f"Background features: {len(features.get('background_context_features', []))}")
        
        print(f"\n🎯 {self.object_name.upper()} Product Features:")
        for feature in features.get('product_specific_visual_features', []):
            print(f"  - {feature['feature_name']}: {feature['variations']}")
        
        print(f"\n🖼️ Background Features:")
        for feature in features.get('background_context_features', []):
            print(f"  - {feature['feature_name']}: {feature['variations']}")
        
        # 4. Save
        if self.save_features(features):
            print(f"\n🎉 {self.model_name.upper()}_{self.object_name.upper()} Feature Discovery completed!")
            print(f"📁 Result file: {self.output_file}")
            return True
        
        return False


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
    
    # Run Feature Discovery
    discovery = ObjectBiasFeatureDiscovery(
        model_name=MODEL_NAME, 
        object_name=OBJECT_NAME,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    success = discovery.run_discovery()
    
    if success:
        print(f"\n✅ Next step: Run {MODEL_NAME}_{OBJECT_NAME} Feature Extraction code")
    else:
        print(f"\n❌ {MODEL_NAME}_{OBJECT_NAME} Feature Discovery failed - please retry")