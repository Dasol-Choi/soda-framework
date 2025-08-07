"""
SODA Model Factory
Factory pattern for creating image generators
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from config.model_configs import ModelConfigs
from config.paths import PATHS
from .image_generator import (
    BaseImageGenerator, 
    GPTImageGenerator, 
    ImagenImageGenerator, 
    ReplicateImageGenerator
)

class ModelFactory:
    """Factory for creating image generators"""
    
    @staticmethod
    def create_generator(
        model_name: str,
        object_name: str,
        csv_file_path: Optional[str] = None,
        **kwargs
    ) -> BaseImageGenerator:
        """
        Create image generator based on model name
        
        Args:
            model_name: Name of the model (gpt, imagen, replicate, etc.)
            object_name: Name of the object to generate images for
            csv_file_path: Path to CSV file with prompts
            **kwargs: Additional arguments for specific generators
            
        Returns:
            Configured image generator instance
        """
        # Use default CSV path if not provided
        if csv_file_path is None:
            csv_file_path = str(PATHS.get_prompt_file_path(object_name))
        
        # Create generator based on model type
        if model_name == 'gpt':
            return GPTImageGenerator(
                object_name=object_name,
                csv_file_path=csv_file_path,
                api_key=kwargs.get('api_key')
            )
        elif model_name == 'imagen':
            return ImagenImageGenerator(
                object_name=object_name,
                csv_file_path=csv_file_path,
                api_key=kwargs.get('api_key')
            )
        elif model_name == 'replicate':
            # Handle replicate sub-model
            replicate_submodel = kwargs.get('replicate_submodel')
            selected_model = kwargs.get('selected_model', '3')  # Default to Imagen 4
            
            if replicate_submodel:
                # Get model configuration
                model_config = ModelConfigs.get_replicate_model_by_name(replicate_submodel)
                if model_config:
                    selected_model = model_config.key
                    # Update folder name based on model category
                    folder_name = model_config.folder_category
                    kwargs['folder_name'] = folder_name
                else:
                    raise ValueError(f"Unknown replicate sub-model: {replicate_submodel}")
            
            generator = ReplicateImageGenerator(
                object_name=object_name,
                csv_file_path=csv_file_path,
                api_token=kwargs.get('api_token'),
                selected_model=selected_model
            )
            
            # Update folder structure if folder_name is provided
            if 'folder_name' in kwargs:
                folder_name = kwargs['folder_name']
                generator.base_save_dir = str(PATHS.get_image_output_dir(folder_name, object_name))
                generator.log_file = str(PATHS.get_log_file_path(folder_name, object_name))
                generator.metadata_file = str(PATHS.get_metadata_file_path(folder_name, object_name))
                generator.create_folder_structure()
            
            return generator
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """Get information about all available models"""
        return {
            'gpt': {
                'name': 'GPT DALL-E',
                'description': 'OpenAI DALL-E image generation',
                'api_key_env': 'OPENAI_API_KEY',
                'sub_models': None
            },
            'imagen': {
                'name': 'Google Imagen',
                'description': 'Google Imagen image generation',
                'api_key_env': 'GOOGLE_API_KEY',
                'sub_models': None
            },
            'replicate': {
                'name': 'Replicate Models',
                'description': 'Various models via Replicate API',
                'api_key_env': 'REPLICATE_API_TOKEN',
                'sub_models': {
                    'sdxl': 'SDXL (Stable Diffusion XL)',
                    'sd35': 'SD 3.5 Large',
                    'imagen4': 'Google Imagen 4',
                    'flux-schnell': 'FLUX Schnell',
                    'flux-dev': 'FLUX Dev',
                    'sd3-medium': 'SD 3 Medium'
                }
            }
        }
    
    @staticmethod
    def get_replicate_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific Replicate model"""
        model_config = ModelConfigs.get_replicate_model_by_name(model_name)
        if model_config:
            return {
                'key': model_config.key,
                'name': model_config.name,
                'model_id': model_config.model_id,
                'cost_per_image': model_config.cost_per_image,
                'description': model_config.description,
                'use_negative': model_config.use_negative,
                'folder_category': model_config.folder_category
            }
        return None 