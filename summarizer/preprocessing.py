"""
Document preprocessing utilities for multi-document summarization.
"""

import re
import nltk
import spacy
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """Represents a document with metadata."""
    text: str
    title: str = ""
    source: str = ""
    sentences: List[str] = None
    chunks: List[str] = None
    
    def __post_init__(self):
        if self.sentences is None:
            self.sentences = self._split_sentences()
        if self.chunks is None:
            self.chunks = self._create_chunks()
    
    def _split_sentences(self) -> List[str]:
        """Split document into sentences."""
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(self.text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        except OSError:
            # Fallback to NLTK if spaCy model not available
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(self.text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def _create_chunks(self, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Create overlapping chunks from the document."""
        chunks = []
        text = self.text
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


class DocumentPreprocessor:
    """Handles document preprocessing for summarization."""
    
    def __init__(self, config=None):
        self.config = config
        self.stop_words = self._load_stop_words()
        self.nlp = self._load_spacy_model()
    
    def _load_stop_words(self) -> set:
        """Load stop words."""
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except:
            return set()
    
    def _load_spacy_model(self):
        """Load spaCy model."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using NLTK fallback.")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def preprocess_documents(self, documents: List[str]) -> List[Document]:
        """Preprocess a list of documents."""
        processed_docs = []
        
        for i, doc_text in enumerate(documents):
            # Clean the text
            cleaned_text = self.clean_text(doc_text)
            
            # Create document object
            doc = Document(
                text=cleaned_text,
                title=f"Document {i+1}",
                source=f"doc_{i+1}"
            )
            
            processed_docs.append(doc)
        
        return processed_docs
    
    def remove_duplicate_sentences(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate sentences across documents."""
        all_sentences = []
        sentence_to_doc = {}
        
        # Collect all sentences with their source documents
        for doc in documents:
            for sentence in doc.sentences:
                all_sentences.append(sentence)
                sentence_to_doc[sentence] = doc
        
        # Find duplicates using TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicate sentences
        duplicate_indices = set()
        threshold = 0.8  # Similarity threshold for duplicates
        
        for i in range(len(similarity_matrix)):
            if i in duplicate_indices:
                continue
            
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    duplicate_indices.add(j)
        
        # Remove duplicates from documents
        unique_sentences = [sent for i, sent in enumerate(all_sentences) if i not in duplicate_indices]
        
        # Reconstruct documents with unique sentences
        for doc in documents:
            doc.sentences = [sent for sent in doc.sentences if sent in unique_sentences]
            doc.text = ' '.join(doc.sentences)
        
        return documents
    
    def rank_sentences_by_importance(self, documents: List[Document]) -> List[Document]:
        """Rank sentences by importance using TF-IDF."""
        all_sentences = []
        sentence_to_doc = {}
        
        # Collect all sentences
        for doc in documents:
            for sentence in doc.sentences:
                all_sentences.append(sentence)
                sentence_to_doc[sentence] = doc
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        
        # Calculate sentence scores (sum of TF-IDF scores)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Rank sentences by score
        sentence_rankings = list(zip(all_sentences, sentence_scores))
        sentence_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Update documents with ranked sentences
        for doc in documents:
            doc_sentences = [(sent, score) for sent, score in sentence_rankings 
                           if sentence_to_doc[sent] == doc]
            doc.sentences = [sent for sent, score in doc_sentences]
        
        return documents
    
    def create_document_summary(self, documents: List[Document], 
                              max_sentences: int = 10) -> str:
        """Create a summary by selecting top sentences."""
        all_sentences = []
        sentence_to_doc = {}
        
        # Collect all sentences with their source
        for doc in documents:
            for sentence in doc.sentences:
                all_sentences.append(sentence)
                sentence_to_doc[sentence] = doc
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Rank sentences
        sentence_rankings = list(zip(all_sentences, sentence_scores))
        sentence_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences ensuring diversity
        selected_sentences = []
        used_docs = set()
        
        for sentence, score in sentence_rankings:
            if len(selected_sentences) >= max_sentences:
                break
            
            doc = sentence_to_doc[sentence]
            if doc not in used_docs or len(used_docs) >= len(documents):
                selected_sentences.append(sentence)
                used_docs.add(doc)
        
        return ' '.join(selected_sentences)
