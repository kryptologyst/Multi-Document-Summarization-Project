"""
Multi-Document Summarization Package

A comprehensive system for summarizing multiple documents using state-of-the-art NLP models.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .models import MultiDocumentSummarizer
from .evaluation import SummarizationEvaluator
from .preprocessing import DocumentPreprocessor, Document
from .config import Config, ModelConfig, PreprocessingConfig, EvaluationConfig

__all__ = [
    'MultiDocumentSummarizer',
    'SummarizationEvaluator', 
    'DocumentPreprocessor',
    'Document',
    'Config',
    'ModelConfig',
    'PreprocessingConfig',
    'EvaluationConfig'
]
