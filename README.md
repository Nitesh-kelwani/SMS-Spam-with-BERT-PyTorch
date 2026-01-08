# 📧 SMS Spam Classification using BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

A binary classification model built with **PyTorch** and **BERT (Bidirectional Encoder Representations from Transformers)** to detect spam SMS messages. This project utilizes **Transfer Learning** to achieve high accuracy on the UCI SMS Spam Collection dataset.

## 📌 Project Overview

SMS spam is a growing problem that requires sophisticated filtering techniques. Traditional methods (like Naive Bayes) often struggle with context. This project leverages the power of the pre-trained `bert-base-uncased` model to understand the semantic context of messages and classify them as either **Ham** (Legitimate) or **Spam**.

**Key Results:**
* **Validation Accuracy:** 96.00%
* **Training Time:** ~2 Epochs to converge

## 📂 Dataset

The dataset used is the **SMS Spam Collection** from the UCI Machine Learning Repository.
* **Source:** [UCI Archive](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
* **Total Samples:** 5,572 messages
* **Class Imbalance:** The original dataset contains ~87% Ham and ~13% Spam.
* **Handling Imbalance:** I performed random downsampling on the majority class (Ham) to create a balanced training set.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP Library:** Hugging Face Transformers (`BertTokenizer`, `BertModel`)
* **Data Manipulation:** Pandas, NumPy
* **Model Evaluation:** Scikit-Learn

## 🧠 Model Architecture

Instead of training a Transformer from scratch, I used a **Feature Extraction** approach:

1.  **BERT Layer:** Loaded `bert-base-uncased`.
2.  **Freezing:** All 12 layers of BERT were **frozen** to preserve pre-trained weights and reduce computational cost.
3.  **Custom Head:** A trainable classifier was added on top of the BERT embeddings:
    * `Linear(768 -> 256)`
    * `ReLU` Activation
    * `Linear(256 -> 1)`
    * `Sigmoid` Activation (for binary probability)

## 📉 Performance

The model was trained for **2 epochs** with a batch size of **64** using the **Adam** optimizer and **BCELoss**.

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **96.00%** |
| **Validation Loss** | 0.12 |

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/bert-spam-classifier.git](https://github.com/your-username/bert-spam-classifier.git)
    cd bert-spam-classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch transformers pandas numpy scikit-learn datasets
    ```

3.  **Run the Notebook:**
    Open `Bert_spam_classification.ipynb` in Jupyter Notebook or Google Colab to run the training pipeline.

## 🧪 Inference Example

```python
# Ham Example
predict(model, "Dear Nitesh, I hope to see you on Monday")
# Output: 'ham'

# Spam Example
predict(model, "Free entry in 2 a wkly comp to win FA Cup final")
# Output: 'spam'
