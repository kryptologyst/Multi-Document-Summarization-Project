"""
Download and cache required models for the summarization system.
"""

import os
import sys
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer
)
import spacy
import nltk


def download_spacy_model():
    """Download spaCy English model."""
    try:
        print("Downloading spaCy English model...")
        spacy.cli.download("en_core_web_sm")
        print("✅ spaCy model downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")


def download_nltk_data():
    """Download required NLTK data."""
    try:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")


def download_transformers_models():
    """Download and cache transformer models."""
    models = [
        ("t5-small", T5ForConditionalGeneration, T5Tokenizer),
        ("facebook/bart-large-cnn", BartForConditionalGeneration, BartTokenizer),
        ("google/pegasus-xsum", PegasusForConditionalGeneration, PegasusTokenizer)
    ]
    
    for model_name, model_class, tokenizer_class in models:
        try:
            print(f"Downloading {model_name}...")
            
            # Download tokenizer
            tokenizer = tokenizer_class.from_pretrained(model_name)
            
            # Download model
            model = model_class.from_pretrained(model_name)
            
            print(f"✅ {model_name} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")


def main():
    """Main download function."""
    print("🚀 Starting model downloads...")
    print("=" * 50)
    
    # Download spaCy model
    download_spacy_model()
    print()
    
    # Download NLTK data
    download_nltk_data()
    print()
    
    # Download transformer models
    download_transformers_models()
    print()
    
    print("=" * 50)
    print("✅ All models downloaded successfully!")
    print("\nYou can now run the summarization system:")
    print("  python main.py --help")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
