"""
SODA Analysis Settings
Configuration for bias analysis parameters
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class AnalysisConfig:
    """Configuration for SODA bias analysis"""
    
    # Feature extraction settings
    feature_extraction_timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    
    # Analysis thresholds
    bias_threshold: float = 0.1
    diversity_threshold: float = 0.05
    confidence_threshold: float = 0.8
    
    # Output settings
    save_plots: bool = True
    save_metadata: bool = True
    verbose: bool = True
    
    # Demographic categories
    demographic_categories: List[str] = None
    
    def __post_init__(self):
        if self.demographic_categories is None:
            self.demographic_categories = [
                "age_young", "age_middle", "age_elderly",
                "gender_male", "gender_female", "gender_non_binary",
                "ethnicity_asian", "ethnicity_black", "ethnicity_white", 
                "ethnicity_hispanic", "ethnicity_middle_eastern"
            ]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'feature_extraction_timeout': self.feature_extraction_timeout,
            'max_retries': self.max_retries,
            'batch_size': self.batch_size,
            'bias_threshold': self.bias_threshold,
            'diversity_threshold': self.diversity_threshold,
            'confidence_threshold': self.confidence_threshold,
            'save_plots': self.save_plots,
            'save_metadata': self.save_metadata,
            'verbose': self.verbose,
            'demographic_categories': self.demographic_categories
        } 