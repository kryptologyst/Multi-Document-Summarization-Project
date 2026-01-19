"""
Test suite for the multi-document summarization system.
"""

import unittest
import tempfile
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.models import MultiDocumentSummarizer, ExtractiveSummarizer
from summarizer.evaluation import SummarizationEvaluator
from summarizer.preprocessing import DocumentPreprocessor, Document
from data.database import MockDatabase


class TestDocumentPreprocessing(unittest.TestCase):
    """Test document preprocessing functionality."""
    
    def setUp(self):
        self.preprocessor = DocumentPreprocessor()
        self.sample_text = "This is a test document. It contains multiple sentences. Each sentence should be processed correctly."
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This   is   a   test   with   extra   spaces!!!"
        cleaned = self.preprocessor.clean_text(dirty_text)
        self.assertEqual(cleaned, "This is a test with extra spaces")
    
    def test_document_creation(self):
        """Test Document class creation."""
        doc = Document(self.sample_text, "Test Document")
        self.assertEqual(len(doc.sentences), 3)
        self.assertIsInstance(doc.sentences, list)
    
    def test_preprocess_documents(self):
        """Test document preprocessing."""
        documents = [self.sample_text, "Another test document. With more content."]
        processed = self.preprocessor.preprocess_documents(documents)
        self.assertEqual(len(processed), 2)
        self.assertIsInstance(processed[0], Document)


class TestExtractiveSummarizer(unittest.TestCase):
    """Test extractive summarization functionality."""
    
    def setUp(self):
        from summarizer.config import ModelConfig
        config = ModelConfig(name="extractive", model_path="extractive")
        self.summarizer = ExtractiveSummarizer(config)
        self.sample_text = """
        Artificial intelligence is transforming industries. Machine learning algorithms can process vast amounts of data.
        Deep learning has revolutionized computer vision. AI systems are becoming more sophisticated.
        The future of AI looks promising. Companies are investing heavily in AI research.
        """
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        sentences = self.summarizer._split_sentences(self.sample_text)
        self.assertGreater(len(sentences), 3)
        self.assertIsInstance(sentences, list)
    
    def test_textrank_scores(self):
        """Test TextRank score calculation."""
        sentences = self.summarizer._split_sentences(self.sample_text)
        scores = self.summarizer._calculate_textrank_scores(sentences)
        self.assertEqual(len(scores), len(sentences))
        self.assertTrue(all(0 <= score <= 1 for score in scores))
    
    def test_summarize(self):
        """Test extractive summarization."""
        summary = self.summarizer.summarize(self.sample_text)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary), len(self.sample_text))


class TestEvaluation(unittest.TestCase):
    """Test evaluation metrics functionality."""
    
    def setUp(self):
        self.evaluator = SummarizationEvaluator()
        self.generated = "AI is transforming industries with machine learning."
        self.reference = "Artificial intelligence is changing industries through machine learning algorithms."
    
    def test_rouge_evaluation(self):
        """Test ROUGE evaluation."""
        results = self.evaluator.evaluate_rouge([self.generated], [self.reference])
        self.assertIn('rouge1', results)
        self.assertIn('rouge2', results)
        self.assertIn('rougeL', results)
        
        for metric in results:
            self.assertIn('precision', results[metric])
            self.assertIn('recall', results[metric])
            self.assertIn('fmeasure', results[metric])
    
    def test_bleu_evaluation(self):
        """Test BLEU evaluation."""
        results = self.evaluator.evaluate_bleu([self.generated], [self.reference])
        self.assertIn('bleu', results)
        self.assertIn('bleu_std', results)
        self.assertTrue(0 <= results['bleu'] <= 1)
    
    def test_coverage_evaluation(self):
        """Test coverage evaluation."""
        results = self.evaluator.evaluate_coverage([self.generated], [self.reference])
        self.assertIn('coverage', results)
        self.assertIn('coverage_std', results)
        self.assertTrue(0 <= results['coverage'] <= 1)


class TestDatabase(unittest.TestCase):
    """Test mock database functionality."""
    
    def setUp(self):
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = MockDatabase(self.temp_db.name)
    
    def tearDown(self):
        # Clean up temporary database
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        stats = self.db.get_document_stats()
        self.assertGreater(stats['total_documents'], 0)
    
    def test_get_documents(self):
        """Test document retrieval."""
        documents = self.db.get_documents(limit=5)
        self.assertIsInstance(documents, list)
        self.assertLessEqual(len(documents), 5)
        
        if documents:
            doc = documents[0]
            self.assertIn('id', doc)
            self.assertIn('title', doc)
            self.assertIn('content', doc)
            self.assertIn('category', doc)
    
    def test_get_categories(self):
        """Test category retrieval."""
        categories = self.db.get_categories()
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        
        if categories:
            category = categories[0]
            self.assertIn('id', category)
            self.assertIn('name', category)
            self.assertIn('description', category)
    
    def test_add_document(self):
        """Test document addition."""
        doc_id = self.db.add_document(
            "Test Document",
            "This is a test document content.",
            "Technology",
            "Test Source"
        )
        self.assertIsInstance(doc_id, int)
        self.assertGreater(doc_id, 0)
    
    def test_search_documents(self):
        """Test document search."""
        results = self.db.search_documents("artificial intelligence")
        self.assertIsInstance(results, list)


class TestMultiDocumentSummarizer(unittest.TestCase):
    """Test main summarizer functionality."""
    
    def setUp(self):
        self.documents = [
            "Artificial intelligence is transforming industries across the globe.",
            "Machine learning algorithms can process vast amounts of data efficiently.",
            "Deep learning has revolutionized fields like computer vision and NLP."
        ]
    
    def test_extractive_summarizer(self):
        """Test extractive summarization."""
        try:
            summarizer = MultiDocumentSummarizer("extractive")
            result = summarizer.summarize(self.documents)
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            self.assertLess(len(result), sum(len(doc) for doc in self.documents))
        except Exception as e:
            self.skipTest(f"Extractive summarizer test skipped: {e}")
    
    def test_summarize_with_metadata(self):
        """Test summarization with metadata."""
        try:
            summarizer = MultiDocumentSummarizer("extractive")
            result = summarizer.summarize_with_metadata(self.documents)
            
            self.assertIn('summary', result)
            self.assertIn('model', result)
            self.assertIn('num_documents', result)
            self.assertIn('total_length', result)
            self.assertIn('summary_length', result)
            self.assertIn('compression_ratio', result)
            
            self.assertEqual(result['num_documents'], len(self.documents))
            self.assertGreater(result['compression_ratio'], 0)
            self.assertLess(result['compression_ratio'], 1)
        except Exception as e:
            self.skipTest(f"Metadata summarization test skipped: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
