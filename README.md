# Multi-Document Summarization Project

A comprehensive multi-document summarization system using state-of-the-art NLP models and techniques.

## Features

- **Multiple Model Support**: T5, BART, Pegasus, and other transformer-based models
- **Hybrid Summarization**: Both extractive and abstractive summarization methods
- **Advanced Preprocessing**: Document chunking, sentence segmentation, and content filtering
- **Evaluation Metrics**: ROUGE, BERTScore, and other summarization quality metrics
- **Web Interface**: Modern Streamlit-based UI for easy interaction
- **Mock Database**: Sample documents for testing and demonstration
- **Configuration Management**: Flexible model and parameter configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Multi-Document-Summarization-Project.git
cd Multi-Document-Summarization-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python -m spacy download en_core_web_sm
python download_models.py
```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Command Line
```bash
python main.py --model bart --documents doc1.txt doc2.txt --output summary.txt
```

### Python API
```python
from summarizer import MultiDocumentSummarizer

summarizer = MultiDocumentSummarizer(model_name="bart")
summary = summarizer.summarize(documents)
```

## Models Supported

- **T5**: Google's Text-to-Text Transfer Transformer
- **BART**: Facebook's Bidirectional and Auto-Regressive Transformer
- **Pegasus**: Google's Pre-training with Extracted Gap-sentences
- **Extractive Methods**: TextRank, LexRank, and other graph-based approaches

## Project Structure

```
├── app.py                 # Streamlit web interface
├── main.py               # Command-line interface
├── summarizer/           # Core summarization modules
│   ├── __init__.py
│   ├── models.py         # Model implementations
│   ├── preprocessing.py  # Document preprocessing
│   ├── evaluation.py     # Evaluation metrics
│   └── config.py         # Configuration management
├── data/                 # Sample documents and database
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
# Multi-Document-Summarization-Project
