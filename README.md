# 🎮 Game Success Analytics

A Machine Learning project that analyzes video game metadata to identify the factors influencing commercial and critical success.

Rather than being a notebook-based project, this repository is being developed as a modular Python application following software engineering best practices, with the long-term goal of deploying the model as a REST API.

---

## 📖 Project Overview

The gaming industry generates enormous amounts of data, including ratings, reviews, genres, publishers, platforms, release dates, and player engagement metrics.

The objective of this project is to analyze historical game data, engineer meaningful features, and build machine learning models capable of predicting a game's success.

The project emphasizes:

- Clean software architecture
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Machine Learning model development
- Model interpretability
- Deployment using FastAPI
- Experiment tracking with MLflow
- Containerization using Docker

---

## 🎯 Project Goals

- Understand the characteristics of successful games
- Identify features that most influence game ratings
- Build a reproducible ML pipeline
- Follow production-oriented coding practices
- Create a portfolio-quality end-to-end ML project

---

## 🛠️ Tech Stack

### Programming

- Python 3.13

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SHAP *(planned)*
- MLflow *(planned)*
- FastAPI *(planned)*

### Tools

- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```text
game-success-analytics/

├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── eda.py
│   └── ...
│
├── tests/
│
├── main.py
└── README.md
```

---

## 🚀 Development Roadmap

### ✅ Sprint 1 — Project Setup

- Modular project structure
- Configuration management
- CSV data loader
- Dataset profiling
- Initial project architecture

---

### ✅ Sprint 2 — Exploratory Data Analysis

- Numerical analysis
- Missing value analysis
- Duplicate analysis
- Correlation analysis
- Categorical feature exploration
- Data visualizations
- EDA report generation

---

### ⏳ Sprint 3 — Data Preprocessing

- Handle missing values
- Remove invalid records
- Standardize data
- Export cleaned dataset

---

### ⏳ Sprint 4 — Feature Engineering

- Date features
- Genre encoding
- Publisher/developer features
- Feature selection

---

### ⏳ Sprint 5 — Machine Learning

- Baseline model
- Model comparison
- Hyperparameter tuning
- Cross-validation

---

### ⏳ Sprint 6 — Model Evaluation

- Feature importance
- Error analysis
- SHAP explanations

---

### ⏳ Sprint 7 — Deployment

- FastAPI REST API
- Model serialization
- Prediction endpoint

---

### ⏳ Sprint 8 — Productionization

- MLflow
- Docker
- Unit tests
- CI-ready project structure

---

## 📊 Current Progress

| Sprint | Status |
|---------|--------|
| Project Setup | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Data Preprocessing | ⏳ In Progress |
| Feature Engineering | ⏳ Pending |
| Machine Learning | ⏳ Pending |
| Deployment | ⏳ Pending |

---

## 🎓 Learning Objectives

This project is being developed to strengthen practical skills in:

- Software Engineering
- Machine Learning
- Data Analysis
- Data Visualization
- Model Deployment
- Production ML workflows

while following industry best practices rather than notebook-centric experimentation.

---

## 📌 Future Improvements

- Docker support
- MLflow experiment tracking
- Automated testing
- CI/CD pipeline
- Cloud deployment
- Interactive dashboard

---

## 👨‍💻 Author

**Shashank Sharma**

Game Programmer | Machine Learning Enthusiast

GitHub: https://github.com/the-metalmilitia
