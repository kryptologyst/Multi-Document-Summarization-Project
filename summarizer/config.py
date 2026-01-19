"""
Configuration management for the multi-document summarization system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os


@dataclass
class ModelConfig:
    """Configuration for summarization models."""
    name: str
    model_path: str
    max_length: int = 512
    min_length: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50


@dataclass
class PreprocessingConfig:
    """Configuration for document preprocessing."""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    remove_stopwords: bool = True
    min_sentence_length: int = 10
    max_sentences_per_doc: int = 50


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    rouge_types: List[str] = None
    use_bertscore: bool = True
    use_bleu: bool = True
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.model_configs = self._load_model_configs()
        self.preprocessing_config = self._load_preprocessing_config()
        self.evaluation_config = self._load_evaluation_config()
    
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations."""
        default_configs = {
            "t5-small": ModelConfig(
                name="t5-small",
                model_path="t5-small",
                max_length=512,
                min_length=50
            ),
            "bart-large-cnn": ModelConfig(
                name="bart-large-cnn",
                model_path="facebook/bart-large-cnn",
                max_length=1024,
                min_length=100
            ),
            "pegasus-xsum": ModelConfig(
                name="pegasus-xsum",
                model_path="google/pegasus-xsum",
                max_length=512,
                min_length=50
            ),
            "extractive": ModelConfig(
                name="extractive",
                model_path="extractive",
                max_length=200,
                min_length=50
            )
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    model_configs = config_data.get("models", {})
                    return {
                        name: ModelConfig(**config) 
                        for name, config in model_configs.items()
                    }
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_configs
    
    def _load_preprocessing_config(self) -> PreprocessingConfig:
        """Load preprocessing configuration."""
        return PreprocessingConfig()
    
    def _load_evaluation_config(self) -> EvaluationConfig:
        """Load evaluation configuration."""
        return EvaluationConfig()
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.model_configs.get(model_name)
    
    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            "models": {
                name: {
                    "name": config.name,
                    "model_path": config.model_path,
                    "max_length": config.max_length,
                    "min_length": config.min_length,
                    "do_sample": config.do_sample,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k
                }
                for name, config in self.model_configs.items()
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
