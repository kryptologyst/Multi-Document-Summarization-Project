# Quick Start Guide

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd 0543_Multi-Document_Summarization

# Run setup script
python setup.py
```

### 2. Quick Demo
```bash
# Run the basic demo
python 0543.py
```

### 3. Web Interface
```bash
# Launch the Streamlit app
streamlit run app.py
```

### 4. Command Line Usage
```bash
# Summarize files
python main.py --model bart-large-cnn --files doc1.txt doc2.txt --output summary.txt

# Use database documents
python main.py --model extractive --database --category Technology

# Evaluate summaries
python main.py --evaluate --generated summary.txt --reference reference.txt
```

## 📋 Available Models

- **BART Large CNN**: Best for news articles and general text
- **T5 Small**: Good balance of quality and speed
- **Pegasus XSum**: Optimized for extreme summarization
- **Extractive**: Fast, no model download required

## 🎯 Key Features

- **Multiple Input Methods**: Manual input, database selection, file upload
- **Advanced Preprocessing**: Document chunking, deduplication, ranking
- **Evaluation Metrics**: ROUGE, BERTScore, BLEU, coverage analysis
- **Web Interface**: User-friendly Streamlit dashboard
- **Database**: Sample documents for testing

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Fails**: Use `extractive` model instead
2. **Memory Issues**: Use smaller models like `t5-small`
3. **Dependencies**: Run `pip install -r requirements.txt`

### Getting Help

- Check the README.md for detailed documentation
- Run tests: `python -m pytest tests/`
- View logs for detailed error messages

## 📊 Example Output

```
📄 Generated Summary:
--------------------------------------
Artificial intelligence is transforming industries through machine learning algorithms that can process vast amounts of data. Deep learning has revolutionized computer vision and natural language processing, enabling sophisticated AI systems.

📊 Summary Statistics:
Model: extractive
Documents: 3
Total Length: 1,234 characters
Summary Length: 234 characters
Compression Ratio: 19%
```
