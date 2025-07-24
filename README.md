# Fake News Detection using Machine Learning

This project focuses on detecting and classifying news articles as either real or fake using Natural Language Processing (NLP) and a supervised machine learning algorithm. It addresses the growing challenge of online misinformation by automating the identification of fake news.

---

## Objective

To build a robust machine learning model that accurately predicts the authenticity of a news article based on its textual content.

---

## Tech Stack

- **Programming Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries and Tools:**  
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn  
- **NLP Technique:** CountVectorizer  
- **ML Algorithm:** PassiveAggressiveClassifier  

---

## Features

- Load and inspect labeled fake/real news articles  
- Perform text cleaning and preprocessing  
- Conduct Exploratory Data Analysis (EDA)  
- Convert textual data into numerical form using CountVectorizer  
- Train a machine learning classification model  
- Evaluate model performance  
- Save the trained model and run predictions on new data  

---

## Dataset

- **Source:** [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- **Details:** Contains two CSV files with labeled news articles categorized as "fake" and "real".

---

## How to Run the Project

Follow the steps below to run the project on your local machine:

1. Install Required Libraries
Ensure that the following Python libraries are installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
2.Launch the Notebook
jupyter notebook fake_news_detection.ipynb
3.Run the Notebook
Inside the notebook interface, select Cell > Run All or run each cell manually. The following steps will be executed:

   -Data loading and inspection
   -Data preprocessing and vectorization
   -Exploratory Data Analysis (EDA)
   -Model training and accuracy evaluation
   -Saving the trained model
   -Testing predictions on new data


