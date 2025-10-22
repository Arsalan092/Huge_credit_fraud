🚨 Fraud Detection using LightGBM
This repository contains a machine learning pipeline for detecting fraudulent transactions. The project leverages LightGBM for classification, SMOTE for handling class imbalance, and precision–recall threshold tuning to optimize fraud detection performance.

✨ Key Highlights:

- 📊 Data Preprocessing
- Frequency encoding for categorical features: type, nameOrig, nameDest.
- Dropped irrelevant columns (isFlaggedFraud).
- ⚖️ Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance fraud vs. non-fraud classes.
- 
- 🤖 Model Training
- LightGBM Classifier with tuned hyperparameters.
- Evaluation metric: F1-score.
- Precision–Recall curve used to select threshold ensuring ≥70% precision.
- 
- 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Imbalanced-learn
