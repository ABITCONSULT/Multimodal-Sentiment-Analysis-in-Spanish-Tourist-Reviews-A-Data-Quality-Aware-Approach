# Multimodal Sentiment Analysis in Spanish Tourist Reviews: A Data Quality-Aware Approach

This repository contains the official implementation, baseline models, and code for the paper **"Multimodal Sentiment Analysis in Spanish Tourist Reviews: A Data Quality-Aware Approach"**, published at **IberLEF 2025**.

This project tackles a multi-task Spanish-language text classification challenge using user-generated tourist reviews from TripAdvisor regarding Mexican destinations (*Pueblos Mágicos*). It implements both traditional machine learning baselines and a unified Transformer architecture designed to handle inherent class imbalances and linguistic complexities.

---

##  Abstract

This work presents baseline approaches for a Spanish-language classification task involving sentiment polarity, entity type, and town identification. We explore traditional machine learning models and pretrained language models like BERT, evaluating their performance across multiple metrics. Despite the linguistic challenges and class imbalances inherent in the dataset, our models provide competitive results and meaningful insights. The analysis highlights the strengths and limitations of each approach, offering a foundation for future improvements in Spanish-language text classification.

---

##  Architecture & Approach

The repository includes implementations for two distinct approaches:

### 1. Unified Transformer Baseline (BETO)

Instead of training separate pipeline stages, we implement a single-encoder architecture that fine-tunes the Spanish whole-word-masking BERT model (`dccuchile/bert-base-spanish-wwm-cased`).

* **Multi-Task Head:** Predicts three targets in parallel via a shared `BertForSequenceClassification` head.
* **Task Framing:** Sentiment polarity is handled as an ordinal classification task, alongside classification for tourist service types and specific town identification.
* **Benefit:** Exploits shared linguistic cues across tasks, significantly reducing inference overhead.

### 2. Traditional Machine Learning Baselines

For benchmarking, we include standard linguistic feature extraction (TF-IDF, word-level embeddings) paired with classic ML classifiers to evaluate performance constraints on class-imbalanced textual data.

---

##  Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Scikit-Learn
* Pandas, NumPy

### Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

```

### Usage

1. **Run Traditional ML Baselines:**
```bash
python train_baselines.py --model random_forest

```



```
2. **Fine-tune the Unified BETO Model:**
   ```bash
   python train_transformer.py --model dccuchile/bert-base-spanish-wwm-cased --epochs 5 --lr 2e-5

```

3. **Evaluate Performance:**
```bash
python evaluate.py --checkpoint ./models/best_beto_shared.pt

```



```

---

##  Key Evaluation Metrics
Models are evaluated across multiple metrics to account for extreme dataset class imbalances:
* Macro F1-Score (Primary metric for balancing underrepresented labels)
* Precision and Recall per subtask
* Comparative analysis of training times vs. unified multi-task inference times

---

##  Citation
If you build upon this work or use the baselines, please cite our IberLEF 2025 paper:

```bibtex
@inproceedings{abiola2025multimodal,
  title={Multimodal Sentiment Analysis in Spanish Tourist Reviews: A Data Quality-Aware Approach},
  author={Abiola, Tolulope Olalekan and Achamaleh, Tewodros and Ojo, Olumide Ebenezer and Adebanji, Olaronke Oluwayemisi and Abiola, Oluwatobi Joseph and Ogunleye, Temitope Dasola and Sidorov, Grigori},
  booktitle={Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2025)},
  series={CEUR Workshop Proceedings},
  year={2025}
}

```

---
