import os
import json
import time
import base64
import pandas as pd
from datetime import datetime
import ast
import random
import requests
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Import required libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Google AI will be imported lazily in ImagenImageGenerator when needed
GOOGLE_AVAILABLE = None  # Will be checked on demand

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

# Global configuration values
IMAGES_PER_PROMPT = 2
DELAY_BETWEEN_REQUESTS = 1  # Delay in seconds
OUTPUT_BASE_DIR = "outputs/images"  # Use images folder

class BaseImageGenerator(ABC):
    """Base class for image generators"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Don't set output_dir here - it will be set in subclasses
        
    @abstractmethod
    def generate_image(self, prompt: str, object_name: str, demographic_info: Optional[Dict] = None) -> str:
        """Generate image and return file path"""
        pass
    
    def setup_output_dir(self, object_name: str):
        """Create output directory for specific object"""
        # This method is deprecated - use base_save_dir instead
        return self.base_save_dir

class GPTImageGenerator(BaseImageGenerator):    
    def __init__(self, object_name: str = None, csv_file_path: str = None, api_key: str = None):
        super().__init__("gpt")
        self.object_name = object_name
        self.csv_file_path = csv_file_path
        
        # GPT-specific configuration
        self.QUALITY = "high"  # "standard" or "hd"
        self.SIZE = "1024x1024"
        
        # Set OpenAI API key
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get from environment variable
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
            else:
                print("Warning: OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        if object_name:
            self.base_save_dir = f"./{OUTPUT_BASE_DIR}/{self.model_name}_{object_name}_images"
            self.log_file = f"{self.base_save_dir}/metadata/generation_log.json"
            self.metadata_file = f"{self.base_save_dir}/metadata/image_metadata.csv"
            self.create_folder_structure()
        
    def create_folder_structure(self):
        folders = [
            f"{self.base_save_dir}/L1_basic",
            f"{self.base_save_dir}/L2_age",
            f"{self.base_save_dir}/L2_gender", 
            f"{self.base_save_dir}/L2_ethnicity",
            f"{self.base_save_dir}/failed_generations",
            f"{self.base_save_dir}/metadata"
        ]
        # First ensure the base output directory exists
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        print(f"📁 {self.object_name} image storage folder structure created")

    def parse_demographic(self, demographic_str):
        """Parse demographic string"""
        if pd.isna(demographic_str) or demographic_str == 'nan' or demographic_str is None:
            return None
        
        try:
            # Convert string to dictionary
            if isinstance(demographic_str, str):
                demographic = ast.literal_eval(demographic_str)
            else:
                demographic = demographic_str
            return demographic
        except:
            return None

    def get_save_path(self, row, image_num):
        """Generate image save path"""
        level = row['level']
        demographic = self.parse_demographic(row.get('demographic'))
        
        # Determine folder
        if level == 1:
            folder_name = "L1_basic"
            target_str = "baseline"
        elif level == 2 and demographic:
            demo_type = demographic.get('type', 'unknown')
            demo_value = demographic.get('value', 'unknown')
            folder_name = f"L2_{demo_type}"
            target_str = demo_value.replace(" ", "_").lower()
        else:
            folder_name = f"L{level}_unknown"
            target_str = "unknown"
        
        folder_path = os.path.join(self.base_save_dir, folder_name)
        
        # Generate filename
        filename_parts = [
            f"L{level}",
            f"idx{image_num:03d}",
            f"img{image_num:02d}",
            target_str,
            datetime.now().strftime("%H%M%S")
        ]
        filename = "_".join(filename_parts) + ".png"
        
        return os.path.join(folder_path, filename)

    def load_progress_log(self):
        """Load progress log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    if "metadata" not in log_data:
                        log_data["metadata"] = []
                    return log_data
            except Exception:
                return {"completed": [], "failed": [], "metadata": []}
        else:
            return {"completed": [], "failed": [], "metadata": []}

    def save_progress_log(self, log_data):
        """Save progress log"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            # Also save to metadata folder
            metadata_log_path = os.path.join(self.base_save_dir, "metadata", "generation_log.json")
            os.makedirs(os.path.dirname(metadata_log_path), exist_ok=True)
            with open(metadata_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Log save failed: {e}")

    def save_image_metadata(self, metadata_list):
        """Save image metadata"""
        try:
            df_meta = pd.DataFrame(metadata_list)
            df_meta.to_csv(self.metadata_file, index=False)
            print(f"📊 Metadata saved: {len(metadata_list)} items")
        except Exception as e:
            print(f"❌ Metadata save failed: {e}")

    def load_prompts(self):
        """Load prompts CSV file"""
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"📄 {len(df)} prompts loaded successfully")
            return df
        except Exception as e:
            print(f"❌ CSV file load failed: {e}")
            return None

    def generate_single_image(self, prompt, output_path, max_retries=3):
        """Generate single image"""
        for attempt in range(max_retries):
            try:
                response = openai.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    n=1,
                    size=self.SIZE,
                    quality=self.QUALITY
                )
                
                # Save image using base64
                image_base64 = response.data[0].b64_json
                
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_base64))
                
                return True, None
                
            except Exception as e:
                print(f"  ⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return False, str(e)

    def generate_all_images(self):
        """Generate all images"""
        df = self.load_prompts()
        if df is None:
            return
        
        log_data = self.load_progress_log()
        completed_set = {(e['prompt_index'], e['image_number']) for e in log_data.get("completed", [])}
        
        total_needed = len(df) * IMAGES_PER_PROMPT
        total_completed = len(completed_set)
        total_generated = 0
        total_failed = 0
        
        print(f"\n🚀 {self.object_name.upper()} image generation started")
        print(f"📊 Total needed: {total_needed}, Completed: {total_completed}")
        print(f"🎯 Images to generate: {total_needed - total_completed}")
        print("-" * 50)
        
        for idx, row in df.iterrows():
            prompt = row['prompt']
            level = row['level']
            demographic = self.parse_demographic(row.get('demographic'))
            
            # Print prompt information
            demo_info = ""
            if demographic:
                demo_type = demographic.get('type', '')
                demo_value = demographic.get('value', '')
                demo_info = f"[{demo_type}: {demo_value}]"
            
            print(f"\n📝 [{idx+1}/{len(df)}] L{level} {demo_info}")
            print(f"   Prompt: {prompt}")
            
            for img_num in range(1, IMAGES_PER_PROMPT + 1):
                # Skip already completed images
                if (idx, img_num) in completed_set:
                    print(f"   ✅ img{img_num:02d} - already completed")
                    continue
                
                output_path = self.get_save_path(row, img_num)
                
                # Skip if file already exists
                if os.path.exists(output_path):
                    print(f"   ✅ img{img_num:02d} - file exists")
                    continue
                
                print(f"   🖼️ img{img_num:02d} generating...", end=" ")
                
                success, error = self.generate_single_image(prompt, output_path)
                
                # Create metadata
                metadata = {
                    "prompt_index": idx,
                    "image_number": img_num,
                    "prompt": prompt,
                    "level": level,
                    "level_name": row.get('level_name', ''),
                    "object": row.get('object', self.object_name),
                    "object_category": row.get('object_category', ''),
                    "demographic_type": demographic.get('type') if demographic else None,
                    "demographic_value": demographic.get('value') if demographic else None,
                    "file_path": output_path,
                    "generation_time": datetime.now().isoformat(),
                    "success": success
                }
                
                if success:
                    total_generated += 1
                    log_data["completed"].append(metadata)
                    log_data["metadata"].append(metadata)
                    print("✅ Success")
                else:
                    total_failed += 1
                    metadata['error'] = error
                    log_data["failed"].append(metadata)
                    print(f"❌ Failed: {error[:50]}")
                
                # Save log
                self.save_progress_log(log_data)
                
                # Control API call interval
                time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Save final metadata
        self.save_image_metadata(log_data["metadata"])
        
        print("\n" + "="*50)
        print(f"🎉 {self.object_name.upper()} image generation completed!")
        print(f"📊 Total generated: {total_generated}")
        print(f"❌ Failed: {total_failed}") 
        print(f"📁 Save location: {self.base_save_dir}")
        print("="*50)
        
        # Return list of generated file paths
        generated_files = []
        for metadata in log_data["metadata"]:
            if metadata.get("success"):
                generated_files.append(metadata.get("file_path"))
        
        return generated_files

    def generate_image(self, prompt: str, object_name: str, demographic_info: Optional[Dict] = None) -> str:
        """Unified interface for single image generation"""
        # Folder structure should already be set up in __init__
        # Skip duplicate folder creation
        pass
        
        # For compatibility with unified interface
        output_path = self.get_save_path({
            'level': 1 if demographic_info is None else 2,
            'demographic': demographic_info,
            'object': object_name
        }, 1)
        
        success, error = self.generate_single_image(prompt, output_path)
        if success:
            return output_path
        else:
            return None

class ImagenImageGenerator(BaseImageGenerator):    
    def __init__(self, object_name: str = None, csv_file_path: str = None, api_key: str = None):
        super().__init__("imagen")
        self.object_name = object_name
        self.csv_file_path = csv_file_path
        
        # Imagen-specific configuration
        self.ASPECT_RATIO = "1:1"
        self.BASE_DELAY = 10  # Base delay increased to 10 seconds
        self.MAX_DELAY = 120  # Maximum delay increased to 2 minutes
        self.BATCH_SIZE = 2  # Generate only 2 at a time then rest
        self.BATCH_DELAY = 30  # Rest time between batches increased to 30 seconds
        
        # Import Google AI libraries only when needed (lazy import)
        try:
            import google.generativeai as genai
            from google.generativeai import types
            self.genai = genai
            self.types = types
        except (ImportError, TypeError) as e:
            raise ImportError(f"Google Generative AI is not available: {e}. Please install google-generativeai>=0.1.0 or use a different model.")
        
        # Set Google API key
        if api_key:
            self.genai.configure(api_key=api_key)
        else:
            # Try to get from environment variable
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.genai.configure(api_key=api_key)
            else:
                print("Warning: Google API key not provided. Please set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        if object_name:
            self.base_save_dir = f"./{OUTPUT_BASE_DIR}/{self.model_name}_{object_name}_images"
            self.log_file = f"{self.base_save_dir}/metadata/generation_log.json"
            self.metadata_file = f"{self.base_save_dir}/metadata/image_metadata.csv"
            self.create_folder_structure()
        
    def create_folder_structure(self):
        """Create folders according to ObjectBias structure"""
        folders = [
            f"{self.base_save_dir}/L1_basic",
            f"{self.base_save_dir}/L2_age",
            f"{self.base_save_dir}/L2_gender", 
            f"{self.base_save_dir}/L2_ethnicity",
            f"{self.base_save_dir}/failed_generations",
            f"{self.base_save_dir}/metadata"
        ]
        # First ensure the base output directory exists
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        print(f"📁 {self.object_name} image storage folder structure created")

    def get_save_path(self, row, image_num):
        """Generate image save path"""
        level = row['level']
        demographic = row.get('demographic')
        if pd.notna(demographic) and demographic != 'nan':
            try:
                demographic = ast.literal_eval(demographic)
            except:
                demographic = None
        
        if level == 1:
            folder_name = "L1_basic"
            target_str = "baseline"
        elif level == 2 and demographic:
            demo_type = demographic.get('type', 'unknown')
            demo_value = demographic.get('value', 'unknown')
            folder_name = f"L2_{demo_type}"
            target_str = demo_value.replace(" ", "_").lower()
        else:
            folder_name = f"L{level}_unknown"
            target_str = "unknown"
        
        folder_path = os.path.join(self.base_save_dir, folder_name)
        
        filename_parts = [
            f"L{level}",
            f"idx{image_num:03d}",
            f"img{image_num:02d}",
            target_str,
            datetime.now().strftime("%H%M%S")
        ]
        filename = "_".join(filename_parts) + ".png"
        return os.path.join(folder_path, filename)

    def save_image_metadata(self, metadata_list):
        """Save image metadata"""
        try:
            df_meta = pd.DataFrame(metadata_list)
            df_meta.to_csv(self.metadata_file, index=False)
            print(f"📊 Metadata saved: {len(metadata_list)} items")
        except Exception as e:
            print(f"❌ Metadata save failed: {e}")

    def load_progress_log(self):
        """Load progress log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    if "metadata" not in log_data:
                        log_data["metadata"] = []
                    if "consecutive_failures" not in log_data:
                        log_data["consecutive_failures"] = 0
                    return log_data
            except Exception:
                return {"completed": [], "failed": [], "metadata": [], "consecutive_failures": 0}
        else:
            return {"completed": [], "failed": [], "metadata": [], "consecutive_failures": 0}

    def save_progress_log(self, log_data):
        """Save progress log"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            # Also save to metadata folder
            metadata_log_path = os.path.join(self.base_save_dir, "metadata", "generation_log.json")
            os.makedirs(os.path.dirname(metadata_log_path), exist_ok=True)
            with open(metadata_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Log save failed: {e}")

    def load_prompts(self, csv_path):
        """Load prompts CSV file"""
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"❌ CSV file load failed: {e}")
            return None

    def adaptive_delay(self, consecutive_failures):
        """Adaptive delay based on consecutive failures"""
        if consecutive_failures == 0:
            return self.BASE_DELAY
        elif consecutive_failures <= 2:
            return self.BASE_DELAY * 2
        elif consecutive_failures <= 4:
            return self.BASE_DELAY * 3
        elif consecutive_failures <= 6:
            return self.BASE_DELAY * 4
        else:
            return min(self.MAX_DELAY, self.BASE_DELAY * 8)

    def generate_single_image(self, prompt, output_path, consecutive_failures=0, max_retries=5):
        """Generate single image"""
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = self.genai.GenerativeModel.generate_images(
                    model='imagen-4.0-generate-preview-06-06',
                    prompt=prompt,
                    config=self.types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=self.ASPECT_RATIO
                    )
                )
                
                if response.generated_images:
                    generated_image = response.generated_images[0]
                    generated_image.image.save(output_path)
                    
                    generation_time = time.time() - start_time
                    return True, None, generation_time
                else:
                    generation_time = time.time() - start_time
                    return False, "No images generated", generation_time
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit related errors
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
                    wait_time = self.adaptive_delay(consecutive_failures) * (2 ** attempt)
                    print(f"  🚫 Rate limit detected! Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}")
                    if attempt < max_retries - 1:
                        print(f"  ⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                
                if attempt == max_retries - 1:
                    generation_time = time.time() - start_time
                    return False, str(e), generation_time

    def generate_all_images(self):
        """Generate all images"""
        df = self.load_prompts(self.csv_file_path)
        if df is None:
            return []
        
        log_data = self.load_progress_log()
        completed_set = {(e['prompt_index'], e['image_number']) for e in log_data.get("completed", [])}
        total_generated, total_failed = 0, 0
        total_generation_time = 0
        overall_start_time = time.time()
        consecutive_failures = log_data.get("consecutive_failures", 0)
        batch_count = 0

        print(f"🚀 Google Imagen-4 image generation started!")
        print(f"📊 Total {len(df)} prompts × {IMAGES_PER_PROMPT} images = {len(df) * IMAGES_PER_PROMPT} images")
        print(f"🎨 Model: imagen-4.0-generate-preview-06-06")
        print(f"📐 Resolution: {self.ASPECT_RATIO} (square)")
        print(f"⏱️ Base delay: {self.BASE_DELAY} seconds, Batch size: {self.BATCH_SIZE}")
        print(f"🔄 Consecutive failures: {consecutive_failures}")
        print("-" * 50)

        for idx, row in df.iterrows():
            prompt = row['prompt']
            print(f"\n📝 [{idx+1}/{len(df)}] {prompt}")
            
            for img_num in range(1, IMAGES_PER_PROMPT + 1):
                if (idx, img_num) in completed_set:
                    print(f"   ✅ img{img_num:02d} - already completed")
                    continue
                    
                output_path = self.get_save_path(row, img_num)
                if os.path.exists(output_path):
                    print(f"   ✅ img{img_num:02d} - file exists")
                    continue
                    
                print(f"   🖼️ img{img_num:02d} generating...", end=" ")
                success, error, generation_time = self.generate_single_image(prompt, output_path, consecutive_failures)
                total_generation_time += generation_time
                batch_count += 1
                
                metadata = {
                    "prompt_index": idx,
                    "image_number": img_num,
                    "prompt": prompt,
                    "level": row['level'],
                    "level_name": row.get('level_name', ''),
                    "object": row.get('object', ''),
                    "demographic": str(row.get('demographic', '')),
                    "file_path": output_path,
                    "generation_time": datetime.now().isoformat(),
                    "generation_duration_seconds": round(generation_time, 2),
                    "model": "imagen-4.0-generate-preview-06-06",
                    "aspect_ratio": self.ASPECT_RATIO,
                    "consecutive_failures": consecutive_failures,
                    "success": success
                }
                
                if success:
                    total_generated += 1
                    consecutive_failures = 0  # Reset on success
                    log_data["completed"].append(metadata)
                    log_data["metadata"].append(metadata)
                    print(f"✅ Success ({generation_time:.1f}s)")
                else:
                    total_failed += 1
                    consecutive_failures += 1
                    metadata['error'] = error
                    log_data["failed"].append(metadata)
                    print(f"❌ Failed: {error[:50]} ({generation_time:.1f}s)")
                
                # Save consecutive failures count
                log_data["consecutive_failures"] = consecutive_failures
                self.save_progress_log(log_data)
                
                # Long rest between batches
                if batch_count % self.BATCH_SIZE == 0:
                    print(f"   💤 Batch completed - {self.BATCH_DELAY}s rest...")
                    time.sleep(self.BATCH_DELAY)
                else:
                    # Normal delay (with random factor)
                    delay = self.adaptive_delay(consecutive_failures) + random.uniform(0, 1)
                    time.sleep(delay)
                
                # Long rest if too many consecutive failures
                if consecutive_failures >= 10:
                    print(f"   😴 {consecutive_failures} consecutive failures - 5 minute rest...")
                    time.sleep(300)  # 5 minute rest

        self.save_image_metadata(log_data["metadata"])
        
        # Calculate time statistics
        overall_elapsed = time.time() - overall_start_time
        avg_time_per_image = total_generation_time / max(total_generated, 1)
        
        print("\n" + "="*50)
        print(f"🎉 Google Imagen-4 image generation completed!")
        print(f"📊 Total generated: {total_generated}")
        print(f"❌ Failed: {total_failed}")
        print(f"🔄 Final consecutive failures: {consecutive_failures}")
        print(f"⏱️ Average time per image: {avg_time_per_image:.1f} seconds")
        print(f"⏱️ Total generation time: {total_generation_time:.1f} seconds")
        print(f"⏱️ Total elapsed time: {overall_elapsed:.1f} seconds")
        print(f"📁 Save location: {self.base_save_dir}")
        print("="*50)
        
        # Return list of generated file paths
        generated_files = []
        for metadata in log_data["metadata"]:
            if metadata.get("success"):
                generated_files.append(metadata.get("file_path"))
        
        return generated_files

    def generate_image(self, prompt: str, object_name: str, demographic_info: Optional[Dict] = None) -> str:
        """Unified interface for single image generation"""
        # For compatibility with unified interface
        output_path = self.get_save_path({
            'level': 1 if demographic_info is None else 2,
            'demographic': demographic_info,
            'object': object_name
        }, 1)
        
        success, error, generation_time = self.generate_single_image(prompt, output_path)
        if success:
            return output_path
        else:
            return None

class ReplicateImageGenerator(BaseImageGenerator):    
    def __init__(self, object_name: str = None, csv_file_path: str = None, api_token: str = None, selected_model: str = "3"):
        super().__init__("replicate")
        self.object_name = object_name
        self.csv_file_path = csv_file_path
        
        # Import model configurations
        from config.model_configs import ModelConfigs
        
        # Get available models from configuration
        self.AVAILABLE_MODELS = {
            key: {
                "name": config.name,
                "model_id": config.model_id,
                "cost_per_image": config.cost_per_image,
                "description": config.description,
                "use_negative": config.use_negative
            }
            for key, config in ModelConfigs.REPLICATE_MODELS.items()
        }
        
        self.selected_model = selected_model
        self.model_info = self.AVAILABLE_MODELS[selected_model]
        
        if object_name:
            self.base_save_dir = f"./{OUTPUT_BASE_DIR}/{self.model_name}_{object_name}_images"
            self.log_file = f"{self.base_save_dir}/metadata/generation_log.json"
            self.metadata_file = f"{self.base_save_dir}/metadata/image_metadata.csv"
            # Don't create folder structure here - it will be updated by model_factory
        else:
            # Initialize with default values to avoid AttributeError
            self.base_save_dir = f"./{OUTPUT_BASE_DIR}/default_images"
            self.log_file = f"{self.base_save_dir}/metadata/generation_log.json"
            self.metadata_file = f"{self.base_save_dir}/metadata/image_metadata.csv"
        
        # Replicate API configuration
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
        else:
            # Try to get from environment variable
            import os
            api_token = os.getenv("REPLICATE_API_TOKEN")
            if api_token:
                os.environ["REPLICATE_API_TOKEN"] = api_token
            else:
                print("Warning: Replicate API token not provided. Please set REPLICATE_API_TOKEN environment variable or pass api_token parameter.")
        
    def create_folder_structure(self):
        folders = [
            f"{self.base_save_dir}/L1_basic",
            f"{self.base_save_dir}/L2_age",
            f"{self.base_save_dir}/L2_gender", 
            f"{self.base_save_dir}/L2_ethnicity",
            f"{self.base_save_dir}/failed_generations",
            f"{self.base_save_dir}/metadata"
        ]
        # First ensure the base output directory exists
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        print(f"📁 {self.object_name} image storage folder structure created")

    def parse_demographic(self, demographic_str):
        """Parse demographic string"""
        if pd.isna(demographic_str) or demographic_str == 'nan' or demographic_str is None:
            return None
        
        try:
            # Convert string to dictionary
            if isinstance(demographic_str, str):
                demographic = ast.literal_eval(demographic_str)
            else:
                demographic = demographic_str
            return demographic
        except:
            return None

    def get_save_path(self, row, image_num):
        """Generate image save path"""
        level = row['level']
        demographic = self.parse_demographic(row.get('demographic'))
        
        # Determine folder
        if level == 1:
            folder_name = "L1_basic"
            target_str = "baseline"
        elif level == 2 and demographic:
            demo_type = demographic.get('type', 'unknown')
            demo_value = demographic.get('value', 'unknown')
            folder_name = f"L2_{demo_type}"
            target_str = demo_value.replace(" ", "_").lower()
        else:
            folder_name = f"L{level}_unknown"
            target_str = "unknown"
        
        folder_path = os.path.join(self.base_save_dir, folder_name)
        
        # Generate filename
        filename_parts = [
            f"L{level}",
            f"idx{image_num:03d}",
            f"img{image_num:02d}",
            target_str,
            datetime.now().strftime("%H%M%S")
        ]
        filename = "_".join(filename_parts) + ".png"
        
        return os.path.join(folder_path, filename)

    def enhance_prompt(self, prompt):
        """Enhance prompt"""
        enhanced = prompt.strip()
        
        # Add basic quality enhancement keywords
        quality_keywords = [
            "high quality", "detailed", "sharp focus", "professional photography",
            "4k", "ultra HD", "crystal clear", "perfect lighting"
        ]
        
        # Add quality keywords if not present in prompt
        if not any(keyword in enhanced.lower() for keyword in quality_keywords):
            enhanced += ", high quality, detailed, sharp focus"
        
        return enhanced

    def get_negative_prompt(self):
        """Generate negative prompt"""
        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "watermark, signature, text, logo, oversaturated, overexposed, "
            "underexposed, low resolution, artifacts, duplicate, error, "
            "missing fingers, extra digit, fewer digits, fused fingers, "
            "mutation, poorly drawn hands, poorly drawn face, mutation, "
            "deformed, extra limbs, extra arms, missing limbs, floating limbs, "
            "disconnected limbs, malformed hands, blur, out of focus, long neck, "
            "long body, mutated hands and fingers, out of frame, double, "
            "two heads, blurred, ugly, disfigured, too many limbs, "
            "deformed, repetitive, black and white, grainy, extra limbs, "
            "bad anatomy, high pass filter, airbrush, portrait, zoom, "
            "soft focus, smooth skin, closeup, deformed, extra limbs, "
            "extra fingers, mutated hands, bad anatomy, bad proportions, "
            "blind, bad eyes, ugly eyes, dead eyes, blur, vignette, "
            "out of shot, out of focus, gaussian, closeup, monochrome, "
            "grainy, noisy, text, writing, watermark, logo, oversaturation, "
            "over saturation, over shadow"
        )
        return negative_prompt

    def get_model_input(self, prompt, negative_prompt, minimal=False):
        """Configure model-specific input parameters"""
        import random
        
        base_input = {
            "prompt": prompt,
        }
        
        # Imagen 4 doesn't use negative prompt
        if not minimal and self.model_info.get('use_negative', True):
            base_input["negative_prompt"] = negative_prompt
        
        # Generate random seed (integer instead of null)
        random_seed = random.randint(0, 999999)
        
        # Model-specific parameters
        if "imagen-4" in self.model_info['model_id'].lower():
            # Google Imagen 4 parameters (simple)
            base_input.update({
                "aspect_ratio": "1:1",  # 1:1, 4:3, 3:4, 16:9, 9:16
                "output_quality": 90    # Quality setting
            })
        elif "sdxl" in self.model_info['model_id'].lower():
            # SDXL parameters
            base_input.update({
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "scheduler": "K_EULER",
                "seed": random_seed
            })
        elif "stable-diffusion-3" in self.model_info['model_id'].lower():
            # SD3/SD3.5 parameters
            base_input.update({
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 28,
                "guidance_scale": 7.0,
                "seed": random_seed
            })
        elif "flux" in self.model_info['model_id'].lower():
            # FLUX parameters
            base_input.update({
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 4 if "schnell" in self.model_info['model_id'] else 50,
                "guidance_scale": 3.5,
                "seed": random_seed
            })
        else:
            # Default parameters
            base_input.update({
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "seed": random_seed
            })
        
        return base_input

    def generate_single_image(self, prompt, output_path, max_retries=3):
        """Generate single image"""
        enhanced_prompt = self.enhance_prompt(prompt)
        
        # Imagen 4 doesn't use negative prompt
        if self.model_info.get('use_negative', True):
            negative_prompt = self.get_negative_prompt()
        else:
            negative_prompt = ""  # Imagen 4 uses empty string
        
        for attempt in range(max_retries):
            try:
                # Configure model-specific input
                model_input = self.get_model_input(enhanced_prompt, negative_prompt)
                
                # Call Replicate
                output = replicate.run(self.model_info['model_id'], input=model_input)
                
                # Handle FileOutput object (returned by Imagen 4, FLUX 2 Pro, etc.)
                if hasattr(output, 'url'):
                    # FileOutput object with url attribute
                    image_url = output.url
                elif hasattr(output, '__str__') and not isinstance(output, (list, str)):
                    # FileOutput object that can be converted to string
                    image_url = str(output)
                elif isinstance(output, list) and len(output) > 0:
                    # List of URLs (older models)
                    image_url = output[0]
                elif isinstance(output, str):
                    # Direct URL string
                    image_url = output
                elif output:
                    # Try to convert to string as last resort
                    image_url = str(output)
                else:
                    return False, "No image generation result"
                
                # Download from image URL
                response = requests.get(image_url, timeout=60)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    return True, None
                else:
                    return False, f"Image download failed: HTTP {response.status_code}"
                    
            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                if attempt == max_retries - 1:
                    return False, error_msg
                print(f"  ⚠️ Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(5)
        
        return False, "Maximum retry attempts exceeded"

    def load_prompts(self):
        """Load prompts CSV file"""
        try:
            csv_path = self.csv_file_path
            if not os.path.exists(csv_path):
                alternative_paths = [
                    csv_path.replace(' ', '_'),
                    csv_path.replace(' ', ''),
                    f"objectbias_data/{self.object_name.replace(' ', '_')}_prompts.csv"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        csv_path = alt_path
                        break
                else:
                    print(f"❌ CSV file not found: {self.csv_file_path}")
                    return None
            
            df = pd.read_csv(csv_path)
            print(f"📋 Prompts loaded: {len(df)} items")
            
            # Sample output
            print(f"\n📋 Prompt samples:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                prompt_preview = str(row.get('prompt', 'N/A'))[:60] + "..."
                level = row.get('level', 'N/A')
                print(f"   [{i+1}] L{level}: {prompt_preview}")
            
            return df
            
        except Exception as e:
            print(f"❌ CSV load failed: {e}")
            return None

    def load_progress_log(self):
        """Load progress log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    if "metadata" not in log_data:
                        log_data["metadata"] = []
                    return log_data
            except Exception:
                return {"completed": [], "failed": [], "metadata": []}
        else:
            return {"completed": [], "failed": [], "metadata": []}

    def save_progress_log(self, log_data):
        """Save progress log"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            # Also save to metadata folder
            metadata_log_path = os.path.join(self.base_save_dir, "metadata", "generation_log.json")
            os.makedirs(os.path.dirname(metadata_log_path), exist_ok=True)
            with open(metadata_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Log save failed: {e}")

    def save_image_metadata(self, metadata_list):
        """Save image metadata"""
        try:
            df_meta = pd.DataFrame(metadata_list)
            df_meta.to_csv(self.metadata_file, index=False)
            print(f"📊 Metadata saved: {len(metadata_list)} items")
        except Exception as e:
            print(f"❌ Metadata save failed: {e}")



    def _generate_single_task(self, task_data):
        """Generate a single image (for parallel processing)"""
        idx, img_num, prompt, row, output_path = task_data
        
        img_start_time = time.time()
        success, error = self.generate_single_image(prompt, output_path)
        generation_time = time.time() - img_start_time
        
        level = row.get('level', 1)
        demographic = self.parse_demographic(row.get('demographic'))
        
        # Create metadata
        metadata = {
            "prompt_index": idx,
            "image_number": img_num,
            "prompt": prompt,
            "level": level,
            "level_name": row.get('level_name', ''),
            "object": row.get('object', self.object_name),
            "object_category": row.get('object_category', ''),
            "demographic_type": demographic.get('type') if demographic else None,
            "demographic_value": demographic.get('value') if demographic else None,
            "file_path": output_path,
            "generation_time": datetime.now().isoformat(),
            "model": self.model_info['name'],
            "model_id": self.model_info['model_id'],
            "generation_duration_seconds": round(generation_time, 2),
            "success": success,
            "error": error if not success else None
        }
        
        return metadata, success

    def generate_all_images(self, num_workers=5):
        """Generate all images with parallel processing"""
        df = self.load_prompts()
        if df is None:
            return []
        
        log_data = self.load_progress_log()
        completed_set = {(e['prompt_index'], e['image_number']) for e in log_data.get("completed", [])}
        
        total_needed = len(df) * IMAGES_PER_PROMPT
        total_completed = len(completed_set)
        remaining_images = total_needed - total_completed
        
        print(f"\n🚀 {self.object_name.upper()} image generation started")
        print(f"🤖 Model: {self.model_info['name']}")
        print(f"⚡ Parallel workers: {num_workers}")
        print(f"📊 Total needed: {total_needed}, Completed: {total_completed}")
        print(f"🎯 Images to generate: {remaining_images}")
        print("-" * 60)
        
        # Prepare tasks
        tasks = []
        for idx, row in df.iterrows():
            try:
                prompt = row['prompt']
                if pd.isna(prompt):
                    continue
                prompt = str(prompt).strip()
                if prompt == "" or prompt.lower() in ['nan', 'none', 'null']:
                    continue
            except:
                continue
            
            for img_num in range(1, IMAGES_PER_PROMPT + 1):
                if (idx, img_num) in completed_set:
                    continue
                
                output_path = self.get_save_path(row, img_num)
                if os.path.exists(output_path):
                    continue
                
                tasks.append((idx, img_num, prompt, row, output_path))
        
        print(f"📋 Tasks to process: {len(tasks)}")
        
        total_generated = 0
        total_failed = 0
        start_time = time.time()
        lock = threading.Lock()
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(self._generate_single_task, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                idx, img_num, prompt, row, output_path = task
                
                try:
                    metadata, success = future.result()
                    
                    with lock:
                        if success:
                            total_generated += 1
                            log_data["completed"].append(metadata)
                            print(f"✅ [{i}/{len(tasks)}] idx{idx+1:03d}_img{img_num:02d} ({total_generated} generated)")
                        else:
                            total_failed += 1
                            log_data["failed"].append(metadata)
                            print(f"❌ [{i}/{len(tasks)}] idx{idx+1:03d}_img{img_num:02d} - {metadata.get('error', 'Unknown error')}")
                        
                        # Save progress every 10 images
                        if (total_generated + total_failed) % 10 == 0:
                            self.save_progress_log(log_data)
                    
                except Exception as e:
                    print(f"❌ [{i}/{len(tasks)}] Task error: {str(e)}")
                    total_failed += 1
        
        # Final save
        self.save_progress_log(log_data)
        
        # Calculate time statistics
        total_time = time.time() - start_time
        total_skipped = 0
        
        print("\n" + "="*60)
        print(f"🎉 {self.object_name.upper()} image generation completed!")
        print(f"📊 Total generated: {total_generated}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️ Skipped: {total_skipped}")
        print(f"⏱️ Total elapsed time: {total_time:.1f} seconds")
        print(f"📁 Save location: {self.base_save_dir}")
        print("="*60)
        
        # Return list of generated file paths
        generated_files = []
        for metadata in log_data["metadata"]:
            if metadata.get("success"):
                generated_files.append(metadata.get("file_path"))
        
        return generated_files

    def generate_image(self, prompt: str, object_name: str, demographic_info: Optional[Dict] = None) -> str:
        """Unified interface for single image generation"""
        # Folder structure should already be set up in __init__
        # Skip duplicate folder creation
        pass
        
        # For compatibility with unified interface
        output_path = self.get_save_path({
            'level': 1 if demographic_info is None else 2,
            'demographic': demographic_info,
            'object': object_name
        }, 1)
        
        success, error = self.generate_single_image(prompt, output_path)
        if success:
            return output_path
        else:
            return None

class UnifiedImageGenerator:
    """Main class to handle all image generation"""
    
    def __init__(self, gpt_api_key: str = None, gemini_api_key: str = None, replicate_api_token: str = None):
        self.generators = {}
        
        # Initialize GPT generator if API key is provided
        if gpt_api_key:
            self.generators["gpt"] = GPTImageGenerator(api_key=gpt_api_key)
        
        # Initialize Imagen generator if API key is provided
        if gemini_api_key:
            try:
                self.generators["imagen"] = ImagenImageGenerator(api_key=gemini_api_key)
            except ImportError:
                print("Warning: Google Generative AI not available: 'type' object is not subscriptable")
        
        # Initialize Replicate generator if API token is provided
        if replicate_api_token:
            self.generators["replicate"] = ReplicateImageGenerator(api_token=replicate_api_token)
    
    def get_generator(self, model_name: str) -> BaseImageGenerator:
        """Get specific generator by model name"""
        if model_name not in self.generators:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.generators.keys())}")
        return self.generators[model_name]
    
    def generate_for_prompts(self, prompts: List[Dict], model_name: str) -> List[str]:
        """Generate images for all prompts using specified model"""
        generator = self.get_generator(model_name)
        generated_files = []
        
        print(f"=== Generating images with {model_name.upper()} ===")
        
        for i, prompt_data in enumerate(prompts, 1):
            prompt_text = prompt_data['prompt']
            object_name = prompt_data['object']
            demographic = prompt_data.get('demographic')
            
            print(f"[{i}/{len(prompts)}] Generating: {prompt_text}")
            
            try:
                filepath = generator.generate_image(prompt_text, object_name, demographic)
                generated_files.append(filepath)
                if filepath:
                    print(f"✓ Generated: {filepath}")
                else:
                    print(f"✗ Failed to generate image")
                    generated_files.append(None)
            except Exception as e:
                print(f"✗ Failed: {e}")
                generated_files.append(None)
            
            # Add delay between requests
            if i < len(prompts):
                time.sleep(1)
        
        return generated_files
    
    def generate_for_objects(self, objects: List[str], model_name: str, prompt_generator=None):
        """Generate images for specific objects"""
        if prompt_generator is None:
            from src.prompt_generator import ObjectBiasPromptGenerator
            prompt_generator = ObjectBiasPromptGenerator(objects)
        
        # Generate prompts first
        prompt_result = prompt_generator.run_for_objects(objects)
        prompts = prompt_result['prompts']
        
        # Generate images
        generated_files = self.generate_for_prompts(prompts, model_name)
        
        return {
            'prompts': prompts,
            'generated_files': generated_files,
            'summary': prompt_result['summary']
        }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate images using different models')
    parser.add_argument('--model', choices=['gpt', 'imagen', 'replicate'], required=True,
                       help='Model to use for image generation')
    parser.add_argument('--replicate-model', choices=['sdxl', 'sd35', 'imagen4', 'flux-schnell', 'flux-dev', 'sd3-medium'], default='imagen4',
                       help='Replicate model selection: sdxl, sd35, imagen4, flux-schnell, flux-dev, sd3-medium')
    parser.add_argument('--objects', nargs='+', required=True,
                       help='Objects to generate images for')
    parser.add_argument('--gpt-key', help='OpenAI API key for GPT')
    parser.add_argument('--gemini-key', help='Google API key for Imagen')
    parser.add_argument('--replicate-token', help='Replicate API token for Replicate models')
    
    args = parser.parse_args()
    
    # Initialize generators
    image_generator = UnifiedImageGenerator(
        gpt_api_key=args.gpt_key,
        gemini_api_key=args.gemini_key,
        replicate_api_token=args.replicate_token
    )
    
    # For replicate model, set the specific model
    if args.model == 'replicate':
        generator = image_generator.get_generator('replicate')
        # Map model names to model keys
        model_mapping = {
            'sdxl': '1',
            'sd35': '2', 
            'imagen4': '3',
            'flux-schnell': '4',
            'flux-dev': '5',
            'sd3-medium': '6'
        }
        model_key = model_mapping[args.replicate_model]
        generator.selected_model = model_key
        generator.model_info = generator.AVAILABLE_MODELS[model_key]
        print(f"🤖 Using Replicate model: {generator.model_info['name']}")
    
    # Generate images
    result = image_generator.generate_for_objects(args.objects, args.model)
    
    print(f"\n=== Generation Complete ===")
    print(f"Model: {args.model}")
    print(f"Objects: {args.objects}")
    print(f"Total prompts: {len(result['prompts'])}")
    print(f"Generated files: {len([f for f in result['generated_files'] if f is not None])}")

if __name__ == "__main__":
    main() 