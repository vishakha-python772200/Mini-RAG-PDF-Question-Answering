# 📚 Mini RAG Model – PDF Question Answering System

This project implements a simple Retrieval-Augmented Generation (RAG) style system that allows users to ask questions from a PDF document.

Instead of keyword matching, it uses embedding-based semantic similarity to retrieve the most relevant sentence.

---

## 🚀 Project Overview

**The system:**

1. Reads a PDF file
2. Splits text into meaningful sentences
3. Converts sentences into embeddings using MiniLM (BERT-based model)
4. Converts user query into embedding
5. Uses cosine similarity to retrieve the best matching answer

---

## 🧠 Tech Stack

- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- Scikit-learn
- NumPy
- PyPDF

### Run the project
python rag_model.py
# Example output
Ask your question: What is Machine Learning?

Best Answer:
Machine learning is a subset of artificial intelligence...

Similarity Score: 0.82
his project helped me understand:

**How embeddings work**

How semantic search powers modern AI systems

Basics of RAG pipeline architecture

Vector similarity using cosine similarity

Data Science Learner | NLP Enthusiast | Building AI Projects 🚀


