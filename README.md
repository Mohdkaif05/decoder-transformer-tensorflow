# MiniGPT — GPT Style Transformer Built From Scratch Using TensorFlow

##  Overview

MiniGPT is a custom decoder-only Transformer (GPT-style language model) implemented completely from scratch using TensorFlow.

The project focuses on understanding and implementing the internal mechanics of modern Large Language Models.

This implementation includes:
- Multi-Head Self Attention
- Positional Encoding
- Decoder Blocks
- Autoregressive Text Generation
- Custom Training Pipeline
- SentencePiece Tokenization

The model was trained on a TinyStories-style dataset to generate short coherent stories.
Dataset link: https://huggingface.co/datasets/roneneldan/TinyStories

---

#  Features

- GPT-style Decoder-Only Transformer
- Built completely from scratch using TensorFlow, NumPy
- Custom Multi-Head Attention implementation
- Positional Encoding
- Masking
- SentencePiece BPE Tokenizer
- Mixed Precision Training (FP16)
- Top-k Sampling
- Temperature Sampling
---

#  Model Architecture

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

#  Components Implemented

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

# ⚙️ Training Details

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

# 📂 Project Structure

```text
minigpt/
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
├── tokenizer/
│   ├── tok.model
│   └── tok.vocab
│
├── saved_model/
│   ├── config.json
│   ├── gpt_epoch_1_2_3_4.weights.h5
│   ├── gpt_epoch_5.weights.h5
│   ├── gpt_epoch_6.weights.h5
│   └── gpt_epoch_7.weights.h5
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Model Evaluation

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
- memorization behavior
- generalization ability

---

# 📈 Training Progress

| Epoch | Final Loss | Observation |
|---|---|---|
| Epoch 1 to 4 | 2.6882 | Basic sentence formation learned |
| Epoch 5 | 2.6246 | Improved coherence and flow |
| Epoch 6 | 2.5307 | Best overall text quality |
| Epoch 7 | 2.5169 | Slight overfitting begins |
| Epoch 8 | 2.4874 | Severe repetition collapse |

---

# 🧠 Epoch-wise Analysis

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

## Epoch 6 — Best Checkpoint ✅

Epoch 6 produced the best balance between:
- coherence
- grammar
- creativity
- stability

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

## Epoch 8 — Model Collapse ❌

Although training loss decreased further, generation quality collapsed completely.

### Failure Pattern
```text
at at at at at at at...
```

### Cause
This is a classic transformer degeneration problem caused by:
- overfitting
- token probability collapse
- reduced diversity

This epoch was not suitable for deployment.

---

# ⚠️ Important Observation

This project demonstrated an important language model training behavior:

> Lower training loss does not always produce better text generation quality.

The best generation quality was achieved before the minimum loss value.

---

# 🏆 Best Model

| Best Checkpoint | Epoch 6 |
|---|---|
| Best Loss-Quality Balance | ✅ |
| Most Coherent Stories | ✅ |
| Best Generalization | ✅ |
| Lowest Repetition | ✅ |

The Epoch 6 checkpoint was selected as the final deployment model.

---

# 📌 Key Learning

This evaluation process helped demonstrate:
- learning progression in transformers
- overfitting behavior
- text degeneration patterns
- relationship between loss and generation quality
- importance of qualitative evaluation in language models

---




Clone repository:

```bash
git clone https://github.com/Mohdkaif05/decoder-transformer-tensorflow

cd minigpt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```text
tensorflow==2.20.0
datasets==4.8.5
os
pickle
numpy==2.2.6
```

---

# 🧠 What I Learned

Through this project I gained practical understanding of:

- Transformer architecture internals
- Attention mechanisms
- Tokenization using SentencePiece
- GPT-style autoregressive generation
- Mixed Precision Training
- TensorFlow custom model building
- Efficient data pipelines
- FastAPI model deployment
- LLM inference workflow

---

# 🚀 Future Improvements

- Train on a larger dataset for better language understanding
- Increase context window for longer story generation
- Experiment with larger transformer architectures
- Add quantization for lightweight deployment
- Build REST API using FastAPI
- Deploy backend model server on Render or Railway
- Build frontend application for interactive story generation
- Deploy frontend using Vercel
- Add real-time text streaming generation
- Create Hugging Face model repository
- Add user-configurable generation parameters
- Build complete end-to-end AI storytelling application

---

# 📊 Current Limitations

- Small model size
- Limited training data
- Short context length
- Occasional repetition in generation
- Training constrained by limited GPU resources (RTX 3050 4GB VRAM)
- Smaller batch sizes due to hardware limitations
- Limited ability to train larger transformer architectures

---

# 🤝 Acknowledgements

Inspired by:
- GPT Architecture
- Attention Is All You Need
- TinyStories Dataset
- Open Source LLM Community

---

# ⭐ Project Goal

The primary goal of this project is educational:
to deeply understand how modern transformer-based language models work internally by implementing them from scratch.

---
