"""
Evaluation metrics for summarization quality.
"""

from typing import List, Dict, Union
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import Counter


class SummarizationEvaluator:
    """Evaluator for summarization quality using multiple metrics."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
    
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate using ROUGE metrics."""
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[metric]['precision'].append(scores[metric].precision)
                rouge_scores[metric]['recall'].append(scores[metric].recall)
                rouge_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
        
        # Calculate averages
        avg_scores = {}
        for metric in rouge_scores:
            avg_scores[metric] = {
                'precision': np.mean(rouge_scores[metric]['precision']),
                'recall': np.mean(rouge_scores[metric]['recall']),
                'fmeasure': np.mean(rouge_scores[metric]['fmeasure'])
            }
        
        return avg_scores
    
    def evaluate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate using BERTScore."""
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore evaluation failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate using BLEU score."""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(
                [ref_tokens], 
                pred_tokens, 
                smoothing_function=self.smoothing_function
            )
            bleu_scores.append(bleu_score)
        
        return {
            'bleu': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores)
        }
    
    def evaluate_coverage(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate coverage of important information."""
        coverage_scores = []
        
        for pred, ref in zip(predictions, references):
            # Extract keywords from reference
            ref_keywords = self._extract_keywords(ref)
            pred_keywords = self._extract_keywords(pred)
            
            # Calculate coverage
            if ref_keywords:
                coverage = len(ref_keywords.intersection(pred_keywords)) / len(ref_keywords)
            else:
                coverage = 0.0
            
            coverage_scores.append(coverage)
        
        return {
            'coverage': np.mean(coverage_scores),
            'coverage_std': np.std(coverage_scores)
        }
    
    def evaluate_redundancy(self, predictions: List[str]) -> Dict[str, float]:
        """Evaluate redundancy in predictions."""
        redundancy_scores = []
        
        for pred in predictions:
            sentences = pred.split('. ')
            if len(sentences) < 2:
                redundancy_scores.append(0.0)
                continue
            
            # Calculate similarity between sentences
            similarities = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = self._calculate_sentence_similarity(sentences[i], sentences[j])
                    similarities.append(sim)
            
            redundancy_scores.append(np.mean(similarities) if similarities else 0.0)
        
        return {
            'redundancy': np.mean(redundancy_scores),
            'redundancy_std': np.std(redundancy_scores)
        }
    
    def evaluate_comprehensive(self, predictions: List[str], references: List[str]) -> Dict[str, Union[Dict, float]]:
        """Comprehensive evaluation using all metrics."""
        results = {}
        
        # ROUGE scores
        results['rouge'] = self.evaluate_rouge(predictions, references)
        
        # BERTScore
        results['bertscore'] = self.evaluate_bertscore(predictions, references)
        
        # BLEU score
        results['bleu'] = self.evaluate_bleu(predictions, references)
        
        # Coverage
        results['coverage'] = self.evaluate_coverage(predictions, references)
        
        # Redundancy
        results['redundancy'] = self.evaluate_redundancy(predictions)
        
        # Overall score (weighted combination)
        rouge_f1 = results['rouge']['rouge1']['fmeasure']
        bertscore_f1 = results['bertscore']['f1']
        bleu_score = results['bleu']['bleu']
        coverage_score = results['coverage']['coverage']
        
        overall_score = (
            0.3 * rouge_f1 +
            0.3 * bertscore_f1 +
            0.2 * bleu_score +
            0.2 * coverage_score
        )
        
        results['overall_score'] = overall_score
        
        return results
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> set:
        """Extract top keywords from text."""
        # Simple keyword extraction using word frequency
        words = text.lower().split()
        
        # Remove stop words
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Return top keywords
        return set([word for word, count in word_counts.most_common(top_k)])
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def print_evaluation_report(self, results: Dict[str, Union[Dict, float]]):
        """Print a formatted evaluation report."""
        print("=" * 60)
        print("SUMMARIZATION EVALUATION REPORT")
        print("=" * 60)
        
        # ROUGE scores
        print("\nROUGE Scores:")
        print("-" * 30)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores = results['rouge'][metric]
            print(f"{metric.upper()}:")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall:    {scores['recall']:.4f}")
            print(f"  F1-Score:  {scores['fmeasure']:.4f}")
        
        # BERTScore
        print(f"\nBERTScore:")
        print("-" * 30)
        bert_scores = results['bertscore']
        print(f"Precision: {bert_scores['precision']:.4f}")
        print(f"Recall:    {bert_scores['recall']:.4f}")
        print(f"F1-Score:  {bert_scores['f1']:.4f}")
        
        # BLEU score
        print(f"\nBLEU Score:")
        print("-" * 30)
        bleu_scores = results['bleu']
        print(f"BLEU:      {bleu_scores['bleu']:.4f}")
        print(f"Std Dev:   {bleu_scores['bleu_std']:.4f}")
        
        # Coverage
        print(f"\nCoverage:")
        print("-" * 30)
        coverage_scores = results['coverage']
        print(f"Coverage:  {coverage_scores['coverage']:.4f}")
        print(f"Std Dev:   {coverage_scores['coverage_std']:.4f}")
        
        # Redundancy
        print(f"\nRedundancy:")
        print("-" * 30)
        redundancy_scores = results['redundancy']
        print(f"Redundancy: {redundancy_scores['redundancy']:.4f}")
        print(f"Std Dev:    {redundancy_scores['redundancy_std']:.4f}")
        
        # Overall score
        print(f"\nOverall Score: {results['overall_score']:.4f}")
        print("=" * 60)
