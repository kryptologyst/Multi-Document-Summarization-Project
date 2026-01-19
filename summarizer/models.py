"""
Advanced summarization models implementation.
"""

import torch
from transformers import (
    pipeline, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from summarizer.config import ModelConfig
from summarizer.preprocessing import Document, DocumentPreprocessor


class BaseSummarizer:
    """Base class for all summarizers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the model and tokenizer."""
        raise NotImplementedError
    
    def summarize(self, text: str) -> str:
        """Generate summary for given text."""
        raise NotImplementedError
    
    def summarize_documents(self, documents: List[str]) -> str:
        """Summarize multiple documents."""
        # Combine documents
        combined_text = " ".join(documents)
        
        # Handle length limitations
        if len(combined_text) > self.config.max_length * 4:  # Rough token estimation
            # Use preprocessing to create a shorter version
            preprocessor = DocumentPreprocessor()
            processed_docs = preprocessor.preprocess_documents(documents)
            processed_docs = preprocessor.remove_duplicate_sentences(processed_docs)
            processed_docs = preprocessor.rank_sentences_by_importance(processed_docs)
            
            # Create summary from top sentences
            summary_text = preprocessor.create_document_summary(processed_docs, max_sentences=20)
            combined_text = summary_text
        
        return self.summarize(combined_text)


class T5Summarizer(BaseSummarizer):
    """T5-based summarizer."""
    
    def load_model(self):
        """Load T5 model and tokenizer."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_path)
        self.model.to(self.device)
    
    def summarize(self, text: str) -> str:
        """Generate summary using T5."""
        if self.model is None:
            self.load_model()
        
        # Prepare input
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.min_length + 50,
                min_length=self.config.min_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary


class BARTSummarizer(BaseSummarizer):
    """BART-based summarizer."""
    
    def load_model(self):
        """Load BART model and tokenizer."""
        self.tokenizer = BartTokenizer.from_pretrained(self.config.model_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_path)
        self.model.to(self.device)
    
    def summarize(self, text: str) -> str:
        """Generate summary using BART."""
        if self.model is None:
            self.load_model()
        
        # Prepare input
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.min_length + 100,
                min_length=self.config.min_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary


class PegasusSummarizer(BaseSummarizer):
    """Pegasus-based summarizer."""
    
    def load_model(self):
        """Load Pegasus model and tokenizer."""
        self.tokenizer = PegasusTokenizer.from_pretrained(self.config.model_path)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.config.model_path)
        self.model.to(self.device)
    
    def summarize(self, text: str) -> str:
        """Generate summary using Pegasus."""
        if self.model is None:
            self.load_model()
        
        # Prepare input
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.min_length + 100,
                min_length=self.config.min_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarization using graph-based methods."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.preprocessor = DocumentPreprocessor()
    
    def load_model(self):
        """No model loading needed for extractive summarization."""
        pass
    
    def summarize(self, text: str) -> str:
        """Generate extractive summary."""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return text
        
        # Calculate sentence scores using TextRank
        sentence_scores = self._calculate_textrank_scores(sentences)
        
        # Select top sentences
        num_sentences = max(3, len(sentences) // 3)
        top_sentences = sorted(
            zip(sentences, sentence_scores),
            key=lambda x: x[1],
            reverse=True
        )[:num_sentences]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        return ' '.join([sent for sent, score in top_sentences])
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import nltk
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def _calculate_textrank_scores(self, sentences: List[str]) -> List[float]:
        """Calculate TextRank scores for sentences."""
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank scores
        scores = nx.pagerank(graph)
        
        return [scores[i] for i in range(len(sentences))]
    
    def summarize_documents(self, documents: List[str]) -> str:
        """Summarize multiple documents using extractive method."""
        # Preprocess documents
        processed_docs = self.preprocessor.preprocess_documents(documents)
        processed_docs = self.preprocessor.remove_duplicate_sentences(processed_docs)
        
        # Combine all sentences
        all_sentences = []
        for doc in processed_docs:
            all_sentences.extend(doc.sentences)
        
        if len(all_sentences) <= 5:
            return ' '.join(all_sentences)
        
        # Calculate scores
        sentence_scores = self._calculate_textrank_scores(all_sentences)
        
        # Select top sentences
        num_sentences = max(5, len(all_sentences) // 4)
        top_sentences = sorted(
            zip(all_sentences, sentence_scores),
            key=lambda x: x[1],
            reverse=True
        )[:num_sentences]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: all_sentences.index(x[0]))
        
        return ' '.join([sent for sent, score in top_sentences])


class MultiDocumentSummarizer:
    """Main class for multi-document summarization."""
    
    def __init__(self, model_name: str = "bart-large-cnn", config_path: str = "config.json"):
        from summarizer.config import Config
        self.config = Config(config_path)
        self.model_config = self.config.get_model_config(model_name)
        
        if self.model_config is None:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        self.summarizer = self._create_summarizer(model_name)
    
    def _create_summarizer(self, model_name: str) -> BaseSummarizer:
        """Create appropriate summarizer based on model name."""
        if model_name.startswith("t5"):
            return T5Summarizer(self.model_config)
        elif model_name.startswith("bart"):
            return BARTSummarizer(self.model_config)
        elif model_name.startswith("pegasus"):
            return PegasusSummarizer(self.model_config)
        elif model_name == "extractive":
            return ExtractiveSummarizer(self.model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def summarize(self, documents: List[str]) -> str:
        """Summarize multiple documents."""
        if not documents:
            return ""
        
        if len(documents) == 1:
            return self.summarizer.summarize(documents[0])
        
        return self.summarizer.summarize_documents(documents)
    
    def summarize_with_metadata(self, documents: List[str]) -> Dict[str, Union[str, Dict]]:
        """Summarize documents and return metadata."""
        summary = self.summarize(documents)
        
        return {
            "summary": summary,
            "model": self.model_config.name,
            "num_documents": len(documents),
            "total_length": sum(len(doc) for doc in documents),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / sum(len(doc) for doc in documents) if documents else 0
        }
