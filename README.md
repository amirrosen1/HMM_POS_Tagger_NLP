# 🧠 HMM POS Tagger – NLP Exercise 3

This repository contains an implementation of several **Part-of-Speech (POS) tagging** models using **Hidden Markov Models (HMMs)** trained on the Brown corpus.

The project was developed as part of an NLP course assignment, and demonstrates different strategies for tagging sequences, handling unknown words, and evaluating performance.

---

## 📚 Project Overview

The goal is to predict the correct POS tag for each word in a sentence.  
To achieve this, we implement and compare multiple taggers:

### ✅ Tagging Methods:

1. **Baseline Tagger**  
   Assigns the **most frequent tag** seen in training for each word.  
   Unknown words are tagged with the overall most common tag.

2. **Bigram HMM Tagger with Viterbi Decoding**  
   Uses a probabilistic sequence model to choose the most likely tag sequence using:
   - Transition probabilities: P(tᵢ | tᵢ₋₁)
   - Emission probabilities: P(wᵢ | tᵢ)

3. **HMM + Laplace Smoothing (Add-One)**  
   Adds smoothing to avoid zero-probability issues when encountering rare/unseen tag transitions.

4. **HMM + Pseudo-Words**  
   Unknown words are mapped to generalized “pseudo-word” categories (e.g., `ALL_CAPS`, `NUMERIC`, `ENDS_WITH_ING`) to improve tagging accuracy.

---

## 📊 Evaluation

- Accuracy is computed on the test set for each tagger
- A **confusion matrix** is produced to analyze common tag mistakes
- The effect of pseudo-word strategies is demonstrated

---

## 📦 Files Included

| File               | Description                                     |
|--------------------|-------------------------------------------------|
| `Ex3_NLP.py`       | Main script – all model implementations         |
| `requirements.txt` | Python dependencies (NLTK, NumPy, etc.)         |
| `README.md`        | This file                                        |
| `.gitignore`       | Excludes large/unnecessary files from the repo  |
| `*.pdf` / `*.docx` | Exercise instructions & solution writeups (ignored) |

---

## 📚 Dataset

We use the **Brown corpus** (news category) from NLTK:

- Downloaded using `nltk.corpus.brown`
- Focused on tagged word sequences for supervised training

---

## ⚙️ Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
