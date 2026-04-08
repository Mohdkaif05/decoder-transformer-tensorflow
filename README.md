# 🚀 Mini GPT from Scratch (TensorFlow)

## 📌 Overview

This project implements a **decoder-only Transformer (GPT-style model)** from scratch using **TensorFlow**, with a strong focus on understanding the underlying mathematics and architecture rather than relying on high-level libraries.

The goal is to build a **clean, modular, and fully interpretable implementation** of a transformer-based language model, covering everything from attention mechanisms to text generation.

---

## 🎯 Objectives

* Build a **GPT-like model from first principles**
* Understand **attention mechanism mathematically and programmatically**
* Implement each component **step-by-step (modular design)**
* Train a small-scale language model on custom datasets
* Generate coherent text using the trained model

---

## 🧠 What This Project Covers

### Core Concepts

* Scaled Dot-Product Attention
* Multi-Head Attention
* Positional Encoding
* Decoder-Only Transformer Architecture
* Masking (Causal / Look-Ahead)
* Feed Forward Networks

### Engineering Aspects

* TensorFlow low-level implementation
* Custom layers using `tf.keras.layers.Layer`
* Training loop with `tf.GradientTape`
* Efficient data pipeline (`tf.data`)
* Text generation (Greedy, Top-k, Temperature sampling)

---

## ⚙️ Tech Stack

* **Framework:** TensorFlow
* **Language:** Python
* **Libraries:** NumPy, Matplotlib (for visualization)

---

## 📂 Project Structure

```bash id="3h1v2k"
mini-gpt-from-scratch/
│
├── notebooks/
│   └── mini_gpt_from_scratch.ipynb
│
├── utils/
│   ├── tokenizer.py
│   ├── dataset.py
│
├── outputs/
│   ├── generated_samples.txt
│   └── checkpoints/
│
└── README.md
```



---

