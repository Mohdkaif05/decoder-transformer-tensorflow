# MiniGPT — GPT-Style Transformer Built From Scratch Using TensorFlow

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
- Dockerized Backend and Frontend
- Cloud Deployment using Render

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
- Dockerized application
- Separate backend and frontend deployment
- Epoch 6 selected as the final deployment checkpoint

---

# Model Architecture

| Parameter | Value |
|---|---:|
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
├── app.py
├── streamlit_app.py
├── requirements.txt
│
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── .dockerignore
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
│   ├── raw/
│   ├── processed/
│   └── training/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── transformer_experiments.ipynb
│
├── README.md
└── .gitignore
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend and MiniGPT inference |
| `streamlit_app.py` | Streamlit frontend |
| `Dockerfile` | Docker image for the FastAPI backend |
| `Dockerfile.streamlit` | Docker image for the Streamlit frontend |
| `docker-compose.yml` | Runs backend and frontend together locally |
| `models/` | Model configuration and Epoch 6 weights |
| `tokenizer/` | SentencePiece tokenizer |
| `requirements.txt` | Python dependencies |

---

# Model Evaluation

The model was trained for multiple epochs and evaluated using both quantitative and qualitative methods.

## Evaluation Criteria

### Quantitative

- Training loss
- Perplexity

### Qualitative

- Story coherence
- Grammar quality
- Repetition control
- Creativity
- Generalization capability

For qualitative evaluation, two prompts were used after every epoch:

1. A prompt similar to the training distribution
2. A prompt not directly seen during training

This helped evaluate both:

- Memorization behavior
- Generalization ability

---

# Model Parameter Summary

The model contains approximately:

| Parameter | Value |
|---|---:|
| Total Parameters | ~7.34 million |
| Trainable Parameters | ~7.34 million |
| Non-trainable Parameters | 0 |
| Model Size | ~27.7 MB |

---

# Loss Analysis

It was not possible to train the model continuously on the laptop due to overheating issues. Therefore, the model was trained one epoch at a time and the loss was recorded after each epoch.

The training loss decreased significantly during the early epochs and then gradually flattened.

| Epoch | Final Loss | Observation |
|---|---:|---|
| Epoch 1–4 | 2.6882 | Basic sentence formation learned |
| Epoch 5 | 2.6246 | Improved coherence and flow |
| Epoch 6 | 2.5307 | Best overall text quality |
| Epoch 7 | 2.5169 | Slight overfitting begins |
| Epoch 8 | 2.4874 | Severe repetition collapse |

The loss continued to decrease through Epoch 8, but generation quality did not improve accordingly.

---

# Perplexity

Perplexity was calculated from the language-model loss using:

```text
Perplexity = exp(Loss)
```

The perplexity decreased substantially during the early stages of training and then stabilized.

The important observation from the evaluation is that **lower perplexity did not necessarily correspond to better generated text**.

Epoch 6 provided the best practical balance between:

- Coherence
- Grammar
- Creativity
- Stability
- Repetition control

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

### Example Problem

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

Although training loss decreased further, generation quality collapsed.

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

Therefore, the final deployed model is **Epoch 6**, rather than the epoch with the lowest training loss.

---

# Best Model

| Evaluation | Result |
|---|---|
| Best Checkpoint | Epoch 6 |
| Best Loss-Quality Balance | Yes |
| Most Coherent Stories | Yes |
| Best Generalization | Yes |
| Lowest Repetition | Yes |

The Epoch 6 checkpoint was selected as the final deployment model.

---

# Application Architecture

The application uses two separately deployed services:

```text
                         User
                          │
                          ▼
                 Streamlit Frontend
                          │
                          │ HTTP POST /generate
                          ▼
                   FastAPI Backend
                          │
                          ▼
                  MiniGPT Transformer
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        SentencePiece  Epoch 6    Autoregressive
         Tokenizer      Weights      Generation
                          │
                          ▼
                Token-by-Token Response
                          │
                          ▼
                 Streamlit Story Display
```

The frontend sends the user's prompt to the FastAPI backend.

The backend loads the SentencePiece tokenizer and Epoch 6 model checkpoint, generates the continuation autoregressively, and streams the generated text progressively back to the frontend.

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
app.py
```

The API provides:

```text
POST /generate
```

Example request:

```json
{
    "prompt": "Once upon a time there was a young boy who loved playing with his toy car."
}
```

The backend validates that the prompt contains at least 50 characters.

The backend is responsible for:

- Loading the MiniGPT model
- Loading the SentencePiece tokenizer
- Receiving prompts
- Autoregressive token generation
- Top-k sampling
- Temperature sampling
- Streaming generated text to the frontend

---

# Frontend

The frontend is implemented using Streamlit.

Location:

```text
streamlit_app.py
```

The frontend provides:

- Story prompt input
- Dynamic character counter
- 50-character minimum validation
- Generate Story button
- Real-time story generation
- Token-by-token story display

The frontend sends requests to the deployed FastAPI backend.

---

# Dockerization

Both application layers are Dockerized separately.

## Backend Dockerization

The backend uses:

```text
Dockerfile
```

The backend Docker image contains:

- Python 3.10
- TensorFlow 2.20.0
- NumPy
- SentencePiece
- FastAPI
- Uvicorn
- MiniGPT model files
- Tokenizer files

The backend container runs FastAPI on:

```text
Port: 8000
```

Example backend Docker command:

```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Frontend Dockerization

The frontend uses:

```text
Dockerfile.streamlit
```

The frontend Docker image contains:

- Python 3.10
- Streamlit
- Requests
- Streamlit application

For Render deployment, Streamlit listens on:

```text
Port: 10000
```

The Dockerfile uses:

```dockerfile
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=10000"]
```

## Docker Compose

The two containers can be run together locally using:

```bash
docker compose build
docker compose up
```

Local architecture:

```text
                    Docker Compose
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
    Streamlit Container       FastAPI Container
         :8501                     :8000
            │                         │
            │      HTTP Request       │
            └────────────────────────►
                                      │
                                      ▼
                               MiniGPT Epoch 6
```

This allows the frontend and backend to be tested locally in an environment similar to the cloud deployment.

---

# Docker Commands

## Build the Services

```bash
docker compose build
```

For a clean rebuild:

```bash
docker compose build --no-cache
```

## Start the Services

```bash
docker compose up
```

## Stop the Services

```bash
docker compose down
```

The Streamlit frontend can be accessed locally at:

```text
http://localhost:8501
```

The FastAPI backend can be accessed locally at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Mohdkaif05/decoder-transformer-tensorflow.git
cd decoder-transformer-tensorflow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Backend Locally

From the project root:

```bash
uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Run Frontend Locally

Open another terminal:

```bash
streamlit run streamlit_app.py
```

The Streamlit application will open in the browser.

---

# Requirements

```text
tensorflow==2.20.0
numpy==2.2.6
sentencepiece
fastapi
uvicorn[standard]
pydantic
streamlit
requests
```

---

# Deployment on Render

The project is deployed on Render as **two separate Dockerized Web Services**.

```text
                         GitHub Repository
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
       Render Frontend Service        Render Backend Service
       Dockerfile.streamlit              Dockerfile
                 │                             │
                 ▼                             ▼
             Streamlit                    FastAPI
                 │                             │
                 │ POST /generate              │
                 └────────────────────────────►
                                               │
                                               ▼
                                        MiniGPT Epoch 6
```

## Live Services

### Frontend

**Streamlit Application**

https://minigpt-frontend.onrender.com

### Backend

**FastAPI API**

https://decoder-transformer-tensorflow.onrender.com

### Backend API Documentation

https://decoder-transformer-tensorflow.onrender.com/docs

---

# Render Frontend Deployment

The frontend is deployed as a separate Render Web Service.

### Configuration

```text
Service Type: Web Service
Runtime: Docker
Dockerfile: ./Dockerfile.streamlit
Port: 10000
```

The frontend Dockerfile starts Streamlit with:

```dockerfile
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=10000"]
```

The Streamlit frontend is publicly available at:

```text
https://minigpt-frontend.onrender.com
```

---

# Render Backend Deployment

The backend is deployed as a separate Render Web Service.

### Configuration

```text
Service Type: Web Service
Runtime: Docker
Dockerfile: ./Dockerfile
Port: 8000
```

The backend provides the story-generation API:

```text
POST /generate
```

The deployed backend is available at:

```text
https://decoder-transformer-tensorflow.onrender.com
```

API documentation:

```text
https://decoder-transformer-tensorflow.onrender.com/docs
```

---

# Frontend–Backend Communication in Production

The production frontend communicates directly with the deployed FastAPI backend.

The frontend API configuration uses:

```python
API_URL = "https://decoder-transformer-tensorflow.onrender.com/generate"
```

Request flow:

```text
User enters prompt
        │
        ▼
Streamlit Frontend
        │
        │ POST /generate
        ▼
FastAPI Backend
        │
        ▼
SentencePiece Tokenizer
        │
        ▼
MiniGPT Epoch 6
        │
        ▼
Autoregressive Generation
        │
        ▼
Streaming Response
        │
        ▼
Streamlit Frontend
        │
        ▼
Generated Story
```

---

# Render Deployment Flow

```text
1. Code is pushed to GitHub
        ↓
2. Render pulls the repository
        ↓
3. Render builds the Docker image
        ↓
4. Docker container starts
        ↓
5. Backend / frontend service becomes available
        ↓
6. Streamlit sends requests to FastAPI
        ↓
7. FastAPI runs MiniGPT inference
        ↓
8. Generated story is streamed back
```

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
- Cloud inference can be slower than local inference because of limited cloud resources

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
- Add stronger evaluation metrics
- Improve production inference performance

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
- Docker containerization
- Multi-container application architecture
- Cloud deployment using Render

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
