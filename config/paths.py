"""
SODA Path Configuration
Centralized path management for the SODA framework
"""

from pathlib import Path
from typing import Dict, Any
import os

class PathConfig:
    """Centralized path configuration for SODA"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    OUTPUTS_DIR = BASE_DIR / "outputs"
    PROMPTS_DIR = BASE_DIR / "prompts"
    ASSETS_DIR = BASE_DIR / "assets"
    
    # Image generation paths
    IMAGES_DIR = OUTPUTS_DIR / "images"
    COMPREHENSIVE_ANALYSIS_DIR = OUTPUTS_DIR / "comprehensive_analysis"
    
    # Analysis subdirectories - organized structure
    BDS_SCORE_DIR = COMPREHENSIVE_ANALYSIS_DIR / "bds_score"
    VAC_SCORES_DIR = COMPREHENSIVE_ANALYSIS_DIR / "vac_scores"
    CDS_SCORE_DIR = COMPREHENSIVE_ANALYSIS_DIR / "cds_score"
    
    # Additional organized directories
    LOGS_DIR = OUTPUTS_DIR / "logs"
    METADATA_DIR = OUTPUTS_DIR / "metadata"
    TEMP_DIR = OUTPUTS_DIR / "temp"
    
    # File patterns
    PROMPT_FILE_PATTERN = "{object}_prompts.csv"
    METADATA_FILE_PATTERN = "image_metadata.csv"
    LOG_FILE_PATTERN = "generation_log.json"
    
    @classmethod
    def get_prompt_file_path(cls, object_name: str) -> Path:
        """Get prompt file path for specific object"""
        return cls.PROMPTS_DIR / cls.PROMPT_FILE_PATTERN.format(object=object_name)
    
    @classmethod
    def get_image_output_dir(cls, model_name: str, object_name: str) -> Path:
        """Get image output directory for model and object"""
        return cls.IMAGES_DIR / f"{model_name}_{object_name}_images"
    
    @classmethod
    def get_metadata_file_path(cls, model_name: str, object_name: str) -> Path:
        """Get metadata file path for model and object"""
        output_dir = cls.get_image_output_dir(model_name, object_name)
        return output_dir / "metadata" / cls.METADATA_FILE_PATTERN
    
    @classmethod
    def get_log_file_path(cls, model_name: str, object_name: str) -> Path:
        """Get log file path for model and object"""
        output_dir = cls.get_image_output_dir(model_name, object_name)
        return output_dir / "metadata" / cls.LOG_FILE_PATTERN
    
    @classmethod
    def get_analysis_file_paths(cls) -> Dict[str, Path]:
        """Get all analysis file paths"""
        return {
            'bds_table': cls.BDS_SCORE_DIR / "baseline_vs_8_demographics_table.csv",
            'vac_matrix': cls.VAC_SCORES_DIR / "vac_matrix.csv",
            'vac_perfect_segregation': cls.VAC_SCORES_DIR / "perfect_segregation_cases.csv",
            'vac_dramatic_shifts': cls.VAC_SCORES_DIR / "dramatic_shifts.csv",
            'cds_matrix': cls.CDS_SCORE_DIR / "cds_feature_diversity_matrix.csv"
        }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.OUTPUTS_DIR,
            cls.PROMPTS_DIR,
            cls.IMAGES_DIR,
            cls.COMPREHENSIVE_ANALYSIS_DIR,
            cls.BDS_SCORE_DIR,
            cls.VAC_SCORES_DIR,
            cls.CDS_SCORE_DIR,
            cls.LOGS_DIR,
            cls.METADATA_DIR,
            cls.TEMP_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_folder_structure(cls, base_dir: Path) -> list:
        """Get standard folder structure for image generation"""
        return [
            base_dir / "L1_basic",
            base_dir / "L2_age",
            base_dir / "L2_gender", 
            base_dir / "L2_ethnicity",
            base_dir / "failed_generations",
            base_dir / "metadata"
        ]
    
    @classmethod
    def get_organized_output_structure(cls) -> Dict[str, Path]:
        """Get organized output directory structure"""
        return {
            'images': cls.IMAGES_DIR,
            'analysis': cls.COMPREHENSIVE_ANALYSIS_DIR,
            'logs': cls.LOGS_DIR,
            'metadata': cls.METADATA_DIR,
            'temp': cls.TEMP_DIR,
            'prompts': cls.PROMPTS_DIR
        }

# Global path configuration instance
PATHS = PathConfig() 