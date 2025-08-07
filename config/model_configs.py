"""
SODA Model Configurations
Configuration for different image generation models
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path

@dataclass
class ModelConfig:
    """Base configuration for image generation models"""
    
    name: str
    api_key_env: str
    max_retries: int = 3
    timeout: int = 30
    quality: str = "high"
    size: str = "1024x1024"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'api_key_env': self.api_key_env,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'quality': self.quality,
            'size': self.size
        }

@dataclass
class ReplicateModelConfig:
    """Configuration for Replicate models"""
    
    key: str
    name: str
    model_id: str
    cost_per_image: float
    description: str
    use_negative: bool = True
    folder_category: str = None
    
    def __post_init__(self):
        if self.folder_category is None:
            # Auto-determine folder category based on model name
            if 'sdxl' in self.name.lower() or 'sd' in self.name.lower():
                self.folder_category = 'diffusion'
            elif 'imagen' in self.name.lower():
                self.folder_category = 'imagen'
            elif 'flux' in self.name.lower():
                self.folder_category = 'flux'
            else:
                self.folder_category = 'other'

class ModelConfigs:
    """Collection of model configurations"""
    
    # Base model configurations
    GPT = ModelConfig(
        name="gpt",
        api_key_env="OPENAI_API_KEY",
        quality="high",
        size="1024x1024"
    )
    
    GEMINI = ModelConfig(
        name="gemini",
        api_key_env="GOOGLE_API_KEY",
        quality="high",
        size="1024x1024"
    )
    
    REPLICATE = ModelConfig(
        name="replicate",
        api_key_env="REPLICATE_API_TOKEN",
        quality="high",
        size="1024x1024"
    )
    
    IMAGEN = ModelConfig(
        name="imagen",
        api_key_env="GOOGLE_API_KEY",
        quality="high",
        size="1024x1024"
    )
    
    # Replicate model configurations
    REPLICATE_MODELS = {
        "1": ReplicateModelConfig(
            key="1",
            name="SDXL (Stable Diffusion XL)",
            model_id="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            cost_per_image=0.0055,
            description="High quality, fast speed, stable",
            use_negative=True,
            folder_category="diffusion"
        ),
        "2": ReplicateModelConfig(
            key="2",
            name="SD 3.5 Large",
            model_id="stability-ai/stable-diffusion-3.5-large:fd89ac5c8e4e5cc7fcc43f17d6ffe8d38e200b2b7833e1b596bb50ee065c07d5",
            cost_per_image=0.055,
            description="Latest model, highest quality, expensive",
            use_negative=True,
            folder_category="diffusion"
        ),
        "3": ReplicateModelConfig(
            key="3",
            name="Google Imagen 4",
            model_id="google/imagen-4",
            cost_per_image=0.04,
            description="🔥 Google's latest model, no negative prompt needed, highest quality",
            use_negative=False,
            folder_category="imagen"
        ),
        "4": ReplicateModelConfig(
            key="4",
            name="FLUX Schnell",
            model_id="black-forest-labs/flux-schnell:bf74d7c88ba1a99bb54fecbc45bfb4ae4a35e5e3cf1c9f5ef0f53fc1c28d96c3",
            cost_per_image=0.003,
            description="Fastest, cheapest",
            use_negative=True,
            folder_category="flux"
        ),
        "5": ReplicateModelConfig(
            key="5",
            name="FLUX Dev",
            model_id="black-forest-labs/flux-dev:71efcdb239752ccad30b2b9de7e0914f9c39b5659bb68b0f2976ffe9d0c8b3d7",
            cost_per_image=0.025,
            description="FLUX high quality version",
            use_negative=True,
            folder_category="flux"
        ),
        "6": ReplicateModelConfig(
            key="6",
            name="SD 3 Medium",
            model_id="stability-ai/stable-diffusion-3:527d2a6296facb8acf55c0c1b2d2d7f46cded3fdd4c0e99de3f35e2c8d0d7c8",
            cost_per_image=0.035,
            description="Balanced performance",
            use_negative=True,
            folder_category="diffusion"
        )
    }
    
    # Model name to key mapping
    REPLICATE_MODEL_MAPPING = {
        'sdxl': '1',
        'sd35': '2', 
        'imagen4': '3',
        'flux-schnell': '4',
        'flux-dev': '5',
        'sd3-medium': '6'
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for specific model"""
        configs = {
            'gpt': cls.GPT,
            'gemini': cls.GEMINI,
            'replicate': cls.REPLICATE,
            'imagen': cls.IMAGEN
        }
        return configs.get(model_name.lower(), cls.GPT)
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, ModelConfig]:
        """Get all model configurations"""
        return {
            'gpt': cls.GPT,
            'gemini': cls.GEMINI,
            'replicate': cls.REPLICATE,
            'imagen': cls.IMAGEN
        }
    
    @classmethod
    def get_replicate_model(cls, model_key: str) -> ReplicateModelConfig:
        """Get Replicate model configuration by key"""
        return cls.REPLICATE_MODELS.get(model_key)
    
    @classmethod
    def get_replicate_model_by_name(cls, model_name: str) -> ReplicateModelConfig:
        """Get Replicate model configuration by name"""
        model_key = cls.REPLICATE_MODEL_MAPPING.get(model_name)
        if model_key:
            return cls.get_replicate_model(model_key)
        return None 