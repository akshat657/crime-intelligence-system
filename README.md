# 🕵️ Crime Intelligence System

**Applied Machine Learning & Data Science Project**

## 📌 Project Overview

The Crime Intelligence System is an end-to-end data science and machine learning project built on an Indian crime dataset (2020–2024).
The goal was to analyze crime patterns and evaluate whether crime case resolution (*Case Closed: Yes/No*) can be **predicted using available attributes at the time of reporting**.

This project emphasizes **realistic data science practices**, including exploratory analysis, leakage prevention, feature engineering, baseline modeling, and honest interpretation of results.

---

## 🎯 Problem Statement

> **Given the details of a reported crime at the time of occurrence, can we predict whether the case will eventually be closed?**

This is framed as a **binary classification problem**.

---

## 📂 Dataset Description

The dataset contains ~40,000 crime records with features such as:

* City
* Crime Description
* Crime Domain
* Weapon Used
* Victim Age & Gender
* Police Deployed
* Date & Time of Occurrence
* Case Closed (target variable)

The target variable is **balanced (~50% Yes / 50% No)**, making the problem non-trivial.

---

## 🔍 Exploratory Data Analysis (EDA) – Key Findings

EDA revealed several important insights:

* No single feature (crime type, city, weapon, police count) strongly predicts case closure.
* Closure rates remain close to 50% across:

  * Crime types
  * Cities
  * Crime domains
  * Police deployment levels
* This indicates that **case resolution depends on complex, weak, multivariate interactions rather than obvious rules**.

These findings motivated the use of machine learning models instead of rule-based logic.

---

## 🛠️ Feature Engineering

A dedicated feature engineering pipeline (`build_features.py`) was implemented with:

* Leakage prevention (removal of post-outcome fields)
* Explicit target encoding
* Time-based features:

  * Report Year
  * Report Month
  * Day of Week
  * Hour of Occurrence
* One-hot encoding for categorical variables
* Robust missing-value handling
* Final model-ready numeric dataset

The pipeline is **deterministic, reusable, and model-agnostic**.

---

## 🤖 Models Evaluated

Three models were implemented and compared:

### 1️⃣ Logistic Regression (Baseline)

* Purpose: Establish a linear baseline
* Result: Performance at chance level (ROC-AUC ≈ 0.49)
* Conclusion: No strong linear relationship exists

### 2️⃣ Decision Tree

* Purpose: Capture non-linear interactions
* Result: Marginal improvement but still near random
* Conclusion: Weak and noisy interaction signals

### 3️⃣ Random Forest

* Purpose: Ensemble learning to amplify weak signals
* Result: Slight or no improvement over Decision Tree
* Conclusion: Limited predictive signal in available features

---

## 📊 Model Performance Summary

| Model               | Accuracy   | ROC-AUC                        |
| ------------------- | ---------- | ------------------------------ |
| Logistic Regression | ~0.49      | ~0.49                          |
| Decision Tree       | ~0.50      | ~0.49                          |
| Random Forest       | ~0.50–0.52 | ~0.50–0.55 (dataset-dependent) |

> **Key Insight:**
> Even advanced ensemble methods failed to significantly outperform random guessing, indicating that the dataset lacks strong predictive information for this task.

---

## 🧠 Final Conclusion (Most Important Part)

This project demonstrates an important real-world data science lesson:

> **Machine learning cannot create signal where none exists.**

Despite correct preprocessing, feature engineering, and multiple models:

* Predictive performance remained near chance
* This suggests that **crime case resolution depends on latent, unobserved, or post-event factors not present in the dataset**

This outcome is **not a failure**, but a scientifically valid and valuable conclusion.

---

## 🏆 What This Project Demonstrates

* End-to-end ML pipeline design
* Data leakage awareness
* Feature engineering best practices
* Model comparison & evaluation
* Honest interpretation of ML results
* Ability to conclude when prediction is *not feasible*

These are **core skills expected from a Data Scientist / ML Engineer**.

---

## 🔧 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* VS Code
* Git & GitHub

---

## 🚀 Future Work

* Reframe as an **analytics problem** instead of prediction
* Analyze factors influencing *delay* in case closure
* City-wise and crime-wise efficiency analysis
* Streamlit dashboard for interactive insights
* Incorporate external data (court load, socio-economic factors)

---

## 👤 Author

**Akshat Khandelwal**
B.Tech CSE | Data Science & AI
GitHub: `akshat657`

---
