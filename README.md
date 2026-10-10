# Document Question Answering System (RAG + NLP)

An NLP project for document processing and question answering.

The project aims to build a Retrieval-Augmented Generation (RAG)
pipeline using semantic search and transformer-based extractive
question answering.

## Current Features

- PDF document loading
- DOCX document loading
- TXT document loading
- File validation and error handling
- Automated testing with Pytest

## Planned Features

- Text chunking
- Sentence Transformers embeddings
- FAISS vector search
- BERT-based extractive question answering
- REST API integration
- Audio input support

## Tech Stack

- Python
- pypdf
- python-docx
- Pytest

Planned:
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Django REST Framework

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/document-qa-rag.git

Create a virtual environment:

python3 -m venv .venv

Activate it:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Testing

Run:

pytest -v

Current test coverage includes document loading,
file validation, and error handling.

Latest local test result: 10 passed.

## Project Status

Work in progress.

Currently implemented: Document Loader.

Next milestone: Text Chunking and Embeddings.
