# MiniGPT — GPT Style Transformer Built From Scratch Using TensorFlow

## Overview

MiniGPT is a custom decoder-only Transformer (GPT-style language model) implemented completely from scratch using TensorFlow.

The project focuses on understanding and implementing the internal mechanics of modern Large Language Models.

This implementation includes:

- Multi-Head Self Attention
- Positional Encoding
- Decoder Blocks
- Autoregressive Text Generation
- Custom Training Pipeline
- SentencePiece Tokenization
- Top-k Sampling
- Temperature Sampling
- FastAPI Inference API
- Streamlit Frontend
- Real-time Token-by-Token Story Generation

The model was trained on a TinyStories-style dataset to generate short coherent stories.

Dataset:

https://huggingface.co/datasets/roneneldan/TinyStories

---

# Features

- GPT-style Decoder-Only Transformer
- Built completely from scratch using TensorFlow and NumPy
- Custom Multi-Head Attention implementation
- Positional Encoding
- Masking
- SentencePiece BPE Tokenizer
- Mixed Precision Training (FP16)
- Top-k Sampling
- Temperature Sampling
- FastAPI Backend
- Streamlit Frontend
- Token-by-Token Streaming Generation
- Epoch 6 selected as the final deployment checkpoint

---

# Model Architecture

| Parameter | Value |
|---|---|
| Architecture | Decoder-Only Transformer |
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Dimension (`d_model`) | 256 |
| Feed Forward Dimension (`dff`) | 1024 |
| Vocabulary Size | 8000 |
| Context Length | 128 |
| Framework | TensorFlow |
| Tokenizer | SentencePiece BPE |

---

# Components Implemented

## Transformer Components

- Scaled Dot Product Attention
- Multi-Head Self Attention
- Positional Encoding
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Decoder Blocks
- Masking

---

# Training Details

| Configuration | Value |
|---|---|
| Dataset | TinyStories-style Dataset |
| Tokenizer | SentencePiece |
| Optimizer | AdamW |
| Precision | Mixed Precision FP16 |
| Loss Function | Sparse Categorical Crossentropy |
| Training Type | Autoregressive Language Modeling |
| Framework | TensorFlow |

---

# Project Structure

```text
minigpt/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── models/
│   ├── config.json
│   └── gpt_epoch_6.weights.h5
│
├── tokenizer/
│   ├── tok.model
│   └── tok.vocab
│
├── data/
│   ├── raw_dataset/
│   │   └── texts_50k.pkl
│   │
│   ├── processed_dataset/
│   │   ├── final_texts_50k.pkl
│   │   └── stories.txt
│   │
│   └── training_data/
│       ├── x_50k.pkl
│       └── y_50k.pkl
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── training.ipynb
│
├── README.md
└── .gitignore
```

---

# Model Evaluation

The model was trained for multiple epochs and evaluated using:

- Training loss
- Story coherence
- Grammar quality
- Repetition control
- Creativity
- Generalization capability

For evaluation, two prompts were used after every epoch:

1. A prompt similar to the training distribution
2. A prompt not directly seen during training

This helped evaluate both:

- Memorization behavior
- Generalization ability

---

# Training Progress

| Epoch | Final Loss | Observation |
|---|---:|---|
| Epoch 1 to 4 | 2.6882 | Basic sentence formation learned |
| Epoch 5 | 2.6246 | Improved coherence and flow |
| Epoch 6 | 2.5307 | Best overall text quality |
| Epoch 7 | 2.5169 | Slight overfitting begins |
| Epoch 8 | 2.4874 | Severe repetition collapse |

---

# Epoch-wise Analysis

## Epoch 1 to 4

### Improvements

- Learned basic sentence structure
- Generated understandable stories
- Began learning character continuity

### Issues

- Semantic confusion
- Weak coherence
- Repetitive structures

### Example Problems

```text
wanted to eat the truck
```

---

## Epoch 5

### Improvements

- Better story flow
- Improved dialogue generation
- More stable text generation

### Issues

- Logical inconsistencies
- Object repetition
- Occasional broken sentences

---

## Epoch 6 — Best Checkpoint

Epoch 6 produced the best balance between:

- Coherence
- Grammar
- Creativity
- Stability

### Improvements

- Better emotional continuity
- Improved narrative progression
- More natural dialogue
- Stronger semantic consistency

### Example

```text
"I'm sorry, Lily. We can fix it together."
```

This checkpoint demonstrated the strongest overall generation quality.

---

## Epoch 7

### Observations

- Slightly lower loss
- Text quality started degrading
- Increased repetitive phrasing

### Signs of Overfitting

```text
dirty and dirty
```

Generation became less creative and more repetitive.

---

## Epoch 8 — Model Collapse

Although training loss decreased further, generation quality collapsed completely.

### Failure Pattern

```text
at at at at at at at...
```

### Cause

This epoch showed severe text degeneration characterized by:

- Overfitting
- Token probability collapse
- Reduced diversity

This epoch was not selected for deployment.

---

# Important Observation

> Lower training loss does not always produce better text generation quality.

The best generation quality was achieved before the minimum loss value.

---

# Best Model

| Best Checkpoint | Epoch 6 |
|---|---|
| Best Loss-Quality Balance | Yes |
| Most Coherent Stories | Yes |
| Best Generalization | Yes |
| Lowest Repetition | Yes |

The Epoch 6 checkpoint was selected as the final deployment model.

---

# Application Architecture

The project contains two application layers:

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ HTTP Streaming Request
 ▼
FastAPI Backend
 │
 ▼
MiniGPT Transformer
 │
 ├── SentencePiece Tokenizer
 ├── Epoch 6 Weights
 └── Autoregressive Generation
 │
 ▼
Token-by-Token Response
 │
 ▼
Streamlit Story Display
```

The frontend sends a story prompt to the FastAPI backend.

The backend generates the story autoregressively and streams the generated text progressively back to the frontend.

---

# Story Generation

The application accepts a text prompt with a minimum length of 50 characters.

Example:

```text
Once upon a time there was a young boy who loved playing with his toy car.
```

The model then generates the continuation token by token.

Generation uses:

- Temperature: `0.9`
- Top-k: `40`
- Maximum new tokens: `120`
- Context length: `128`

---

# Backend

The backend is implemented using FastAPI.

Location:

```text
backend/app.py
```

The API provides a story generation endpoint:

```text
POST /generate
```

Request:

```json
{
    "prompt": "Once upon a time there was a young boy who loved playing with his toy car."
}
```

The backend validates that the prompt contains at least 50 characters.

---

# Frontend

The frontend is implemented using Streamlit.

Location:

```text
frontend/streamlit_app.py
```

The frontend provides:

- Story prompt input
- Dynamic character counter
- 50-character minimum validation
- Generate Story button
- Real-time story generation
- Token-by-token story display

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Mohdkaif05/decoder-transformer-tensorflow.git

cd decoder-transformer-tensorflow
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

---

# Run Backend

From the project root:

```bash
uvicorn backend.app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

# Run Frontend

Open another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit application will open in the browser.

---

# Requirements

## Backend

```text
tensorflow==2.20.0
numpy==2.2.6
sentencepiece
fastapi
uvicorn
```

## Frontend

```text
streamlit
requests
```

---

# Deployment

The application is designed to be deployed using a free-tier cloud setup.

Recommended architecture:

```text
GitHub Repository
       │
       ├── FastAPI Backend
       │      │
       │      └── MiniGPT Model
       │
       └── Streamlit Frontend
              │
              └── Calls FastAPI API
```

The backend hosts:

- Transformer model
- Tokenizer
- Story generation API

The frontend hosts:

- Streamlit interface
- Prompt input
- Character counter
- Generated story display

The Epoch 6 model checkpoint is used for production inference.

---

# Current Limitations

- Small model size
- Limited training data
- Short context length
- Occasional repetition in generation
- Training constrained by limited GPU resources (RTX 3050 4GB VRAM)
- Smaller batch sizes due to hardware limitations
- Limited ability to train larger transformer architectures
- Generation quality is limited compared with large pretrained language models

---

# Future Improvements

- Train on a larger dataset for better language understanding
- Increase context window for longer story generation
- Experiment with larger transformer architectures
- Add quantization for lightweight deployment
- Add user-configurable generation parameters
- Create Hugging Face model repository
- Improve repetition control
- Improve generation quality with better sampling strategies

---

# What I Learned

Through this project I gained practical understanding of:

- Transformer architecture internals
- Attention mechanisms
- Tokenization using SentencePiece
- GPT-style autoregressive generation
- Mixed Precision Training
- TensorFlow custom model building
- Efficient data pipelines
- FastAPI model deployment
- REST API development
- LLM inference workflow
- Real-time token streaming
- Frontend-backend integration

---

# Acknowledgements

Inspired by:

- GPT Architecture
- Attention Is All You Need
- TinyStories Dataset
- Open Source LLM Community

---

# Project Goal

The primary goal of this project is educational:

> To deeply understand how modern transformer-based language models work internally by implementing them from scratch.
