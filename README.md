# 🚦 Road Accident Prevention Using Meta Ensemble Method

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-orange?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-f7931e?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# 📌 Project Overview

Road accidents remain one of the leading causes of injuries and fatalities worldwide.  
This project introduces an intelligent **Road Accident Prevention System** powered by a **Meta Ensemble Learning Framework** to predict accident risk accurately and efficiently.

The model combines multiple machine learning and deep learning algorithms to enhance prediction performance and reliability.

---

# 🎯 Objectives

- Predict potential road accident occurrence based on traffic/environmental factors
- Improve prediction accuracy using ensemble methods
- Provide an interactive and user-friendly prediction dashboard
- Support future smart traffic management systems

---

# 🧠 Model Architecture

This project uses a **Stacked Meta Ensemble Approach**:

## 🔹 Base Models

- **FT Transformer** (Deep Learning Model)
- **Random Forest Classifier**
- **XGBoost Classifier**

## 🔹 Final Meta Model

- **Meta Classifier** combines predictions from all base models for final output.

---

# ⚙️ Tech Stack

| Category | Technologies |
|--------|-------------|
| Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Boosting | XGBoost |
| Data Handling | Pandas, NumPy |
| Model Serialization | Joblib |

---

# 📂 Project Structure

```text
Road_Accident_prevention_using_MetaEnsemble_method/
│── app.py
│── predict.py
│── model.py
│── PREPROCESS_code.py
│── dataset_traffic_accident_prediction1.csv
│── ft_transformer.pth
│── random_forest.pkl
│── xgboost.pkl
│── meta_classifier.pkl
│── feature_names.pkl
│── requirements.txt
│── README.md
# 🚀 Installation Guide
```

# 1️⃣ Clone Repository

```bash
git clone <your-repository-link>
cd Road_Accident_prevention_using_MetaEnsemble_method
```
# Requirments
```text
pip install -r requirements.txt
```
# To Run App
```bash
python -m streamlit run app.py
http://localhost:8502/
```
# 📊 Features
```text
✅ Real-time accident risk prediction
✅ Interactive Streamlit user interface
✅ High accuracy ensemble model
✅ Combines Machine Learning + Deep Learning models
✅ Lightweight and fast deployment
✅ Scalable for smart city applications
```

# 🔍 Prediction Workflow
```text
User enters traffic/environment details
Data preprocessing is applied
Base models generate predictions
Meta classifier combines outputs
Final accident risk result is displayed
```

# 📈 Future Enhancements
```text
Live traffic API integration
Weather-based accident risk prediction
GPS-based hotspot detection
Smart alert notification system
Cloud deployment for public access
```

# 👨‍💻 Developed By
John Nathanael J
Vivek S
Abdul Hashim S
# 📜 License

This project is developed for academic and research purposes
