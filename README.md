# Machine-Learning-Project
ML project ( Cardiotocography (CTG) Dataset — Fetal State Classification ) model

Cardiotocography (CTG) Analysis & Deployment
This project focuses on the end-to-end machine learning pipeline for analyzing Cardiotocographic data to predict fetal health states. The workflow covers everything from exploratory data analysis (EDA) and model optimization in Jupyter Notebooks to deploying a live interactive web application using Streamlit.

🚀 Project Overview
The primary goal is to classify fetal health into three categories (Normal, Suspect, or Pathologic) using the CTG.xls dataset. The project is divided into three main phases:

Interactive Experimentation: Model development and evaluation in Jupyter.

UI Development: Building a user-friendly interface with Streamlit.

Deployment Readiness: Preparing dependency files for cloud hosting.

📂 Project Structure
Plaintext
.
├── ml_env/                # Virtual environment (ignored by Git)
├── model.ipynb            # Jupyter Notebook for experimentation
├── streamlit_app.py       # Streamlit application source code
├── algo_best_model.pkl    # Serialized/Trained model file
├── requirements.txt       # Project dependencies for deployment
├── CTG.xls                # Source dataset
└── model_performance.json # Evaluation metrics output
🛠️ Phase 1: Model Development (model.ipynb)
1. Data Preprocessing & EDA
Data Loading: Importing dataset using pandas and xlrd.

Cleaning: Handling missing values, encoding categorical variables, and feature scaling.

Analysis: Visualizing distributions and handling outliers and multicollinearity to ensure data quality.

Split: Partitioning data into 80/20 training and testing sets.

2. Training & Evaluation
Model Selection: Implementation of multiple algorithms including Linear/Logistic Regression, Decision Trees/Random Forests, and SVM/KNN.

Metrics: Evaluation via Accuracy, Precision, Recall, and F1-Score (Classification) or MAE/MSE (Regression).

Validation: Implementation of K-Fold and Stratified K-Fold Cross-Validation to ensure model robustness.

3. Optimization & Serialization
Hyperparameter Tuning: Utilizing Grid Search and Random Search to find optimal model parameters.

Persistence: Saving the best performing model as a .pkl file using pickle or joblib.

💻 Phase 2: User Interface (streamlit_app.py)
The application provides a web-based dashboard for real-time predictions:

Model Loading: Automatically loads the algo_best_model.pkl on startup.

User Input: Interactive sliders, text inputs, and select boxes for entering fetal heart rate (FHR) features.

Inference: Immediate display of health classification results using st.success() and st.write().

☁️ Phase 3: Deployment Preparation
To ensure the application runs seamlessly in a cloud environment:

Requirements: A manually curated requirements.txt file ensures only necessary dependencies (and their specific versions) are installed.

Verification: All preprocessing steps used during training are mirrored in the Streamlit app to maintain consistency.
