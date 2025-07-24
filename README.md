# Fake News Detection using Machine Learning

This project aims to identify and classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) and machine learning techniques. It uses a dataset of news articles and applies text preprocessing, TF-IDF vectorization, and classification through a PassiveAggressiveClassifier model.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Tech Stack](#tech-stack)  
- [How It Works](#how-it-works)  
- [Model Performance](#model-performance)  
- [How to Run](#how-to-run)  
- [Project Structure](#project-structure)  
- [Conclusion](#conclusion)  
- [License](#license)

---

## Project Overview

Fake news is a serious threat to credibility in media. This project demonstrates how machine learning and NLP can be leveraged to detect fake news with high accuracy. The model is trained on a labeled dataset and can predict whether a given article is real or fake.

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Model Used:** PassiveAggressiveClassifier  
- **Vectorizer:** TF-IDF  
- **Platform:** Jupyter Notebook  

---

## How It Works

1. **Data Cleaning & Preprocessing:**  
   - Removal of punctuation, stopwords, digits  
   - Lowercasing and tokenization  
   - Lemmatization

2. **Feature Extraction:**  
   - TF-IDF vectorizer is used to convert text into numeric vectors.

3. **Model Building:**  
   - A PassiveAggressiveClassifier is trained on the TF-IDF features.

4. **Evaluation:**  
   - Accuracy Score, Confusion Matrix, Classification Report

5. **Deployment Ready:**  
   - Model and vectorizer are saved using `pickle`.

---

## Model Performance

- **Accuracy:** ~93%  
- **Classifier:** PassiveAggressiveClassifier  
- **Evaluation Metrics Used:**  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-Score)

---

## How to Run

1. **Install Required Libraries:**

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
2. **Launch the Notebook:**
   -jupyter notebook fake_news_detection.ipynb
3. Run All Cells to View:
   -Data preprocessing
   -Exploratory Data Analysis (EDA)
   -Model training and predictions
   -Save model and test predictions


## Author

S. Vyshnavi  
Email:sukhavasivyshnavi17@gmail.com

