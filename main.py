"""
Command-line interface for multi-document summarization.
"""

import argparse
import sys
import os
from typing import List
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.models import MultiDocumentSummarizer
from summarizer.evaluation import SummarizationEvaluator
from data.database import MockDatabase


def load_documents_from_files(file_paths: List[str]) -> List[str]:
    """Load documents from text files."""
    documents = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                else:
                    print(f"Warning: Empty file {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return documents


def load_documents_from_database(db_path: str, category: str = None, limit: int = 10) -> List[str]:
    """Load documents from the mock database."""
    try:
        db = MockDatabase(db_path)
        db_docs = db.get_documents(category=category, limit=limit)
        return [doc['content'] for doc in db_docs]
    except Exception as e:
        print(f"Error loading from database: {e}")
        return []


def save_summary(summary: str, output_path: str):
    """Save summary to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {output_path}")
    except Exception as e:
        print(f"Error saving summary: {e}")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Document Summarization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize files using BART model
  python main.py --model bart-large-cnn --files doc1.txt doc2.txt --output summary.txt
  
  # Summarize documents from database
  python main.py --model t5-small --database --category Technology --limit 5
  
  # Evaluate summary quality
  python main.py --evaluate --generated summary.txt --reference reference.txt
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        choices=['bart-large-cnn', 't5-small', 'pegasus-xsum', 'extractive'],
        default='bart-large-cnn',
        help='Summarization model to use'
    )
    
    # Input options
    parser.add_argument(
        '--files', 
        nargs='+', 
        help='Input text files to summarize'
    )
    
    parser.add_argument(
        '--database', 
        action='store_true',
        help='Load documents from mock database'
    )
    
    parser.add_argument(
        '--category', 
        help='Filter database documents by category'
    )
    
    parser.add_argument(
        '--limit', 
        type=int, 
        default=10,
        help='Maximum number of documents to load from database'
    )
    
    # Output options
    parser.add_argument(
        '--output', 
        help='Output file for summary'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    # Evaluation options
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluate summary quality'
    )
    
    parser.add_argument(
        '--generated', 
        help='Path to generated summary file'
    )
    
    parser.add_argument(
        '--reference', 
        help='Path to reference summary file'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        default='config.json',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Handle evaluation mode
    if args.evaluate:
        if not args.generated or not args.reference:
            print("Error: --generated and --reference are required for evaluation")
            sys.exit(1)
        
        try:
            # Load summaries
            with open(args.generated, 'r', encoding='utf-8') as f:
                generated_summary = f.read().strip()
            
            with open(args.reference, 'r', encoding='utf-8') as f:
                reference_summary = f.read().strip()
            
            # Evaluate
            evaluator = SummarizationEvaluator()
            results = evaluator.evaluate_comprehensive([generated_summary], [reference_summary])
            
            # Print results
            evaluator.print_evaluation_report(results)
            
            # Save detailed results if output specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                print(f"Detailed results saved to {args.output}")
        
        except Exception as e:
            print(f"Evaluation error: {e}")
            sys.exit(1)
        
        return
    
    # Load documents
    documents = []
    
    if args.files:
        documents.extend(load_documents_from_files(args.files))
    
    if args.database:
        db_documents = load_documents_from_database(
            "data/sample_documents.db", 
            args.category, 
            args.limit
        )
        documents.extend(db_documents)
    
    if not documents:
        print("Error: No documents to summarize")
        print("Use --files to specify input files or --database to load from database")
        sys.exit(1)
    
    if args.verbose:
        print(f"Loaded {len(documents)} documents")
        for i, doc in enumerate(documents):
            print(f"Document {i+1}: {len(doc)} characters")
    
    # Initialize summarizer
    try:
        print(f"Loading {args.model} model...")
        summarizer = MultiDocumentSummarizer(args.model, args.config)
        
        if args.verbose:
            print(f"Model loaded: {summarizer.model_config.name}")
            print(f"Max length: {summarizer.model_config.max_length}")
            print(f"Min length: {summarizer.model_config.min_length}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generate summary
    try:
        print("Generating summary...")
        result = summarizer.summarize_with_metadata(documents)
        
        # Print summary
        print("\n" + "="*60)
        print("GENERATED SUMMARY")
        print("="*60)
        print(result['summary'])
        print("="*60)
        
        # Print metadata
        print(f"\nModel: {result['model']}")
        print(f"Documents: {result['num_documents']}")
        print(f"Total length: {result['total_length']:,} characters")
        print(f"Summary length: {result['summary_length']:,} characters")
        print(f"Compression ratio: {result['compression_ratio']:.2%}")
        
        # Save to file if specified
        if args.output:
            save_summary(result['summary'], args.output)
        
        # Save metadata if verbose
        if args.verbose and args.output:
            metadata_path = args.output.replace('.txt', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Metadata saved to {metadata_path}")
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
