#  Breast Cancer Classification with Neural Network

![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![Keras](https://img.shields.io/badge/Keras-Neural--Network-green)  

---

##  Problem Definition & Understanding
Breast cancer is one of the most common types of cancer worldwide, and early detection is crucial for effective treatment.  
This project focuses on **developing a predictive model** to classify breast cancer tumors as **benign** (non-cancerous) or **malignant** (cancerous) using supervised machine learning with a **Neural Network**.  

**Goal:**  
- Build a classification model using neural networks.  
- Evaluate model performance on real-world data.  
- Provide interpretable insights to support medical decision-making.  

---

##  Data Collection & Cleaning
### Dataset:
- **Source:** `Breast Cancer Wisconsin (Diagnostic) Dataset` from `sklearn.datasets`  
- **Features:**  
  - 30 numeric features describing tumor characteristics (e.g., radius, texture, smoothness, concavity, symmetry).  
  - Target:  
    - `0` = Malignant  
    - `1` = Benign  

### Preprocessing Steps:
1. **Loaded dataset** from `sklearn.datasets`.  
2. **Checked for missing/null values** → None present.  
3. **Normalized features** using `StandardScaler` for better convergence.  
4. **Split dataset** into:  
   - Training (70%)  
   - Validation (15%)  
   - Test (15%)  

---

##  Data Analysis & Charts
Exploratory Data Analysis (EDA) was performed to understand feature distribution and class balance.  

- **Class Distribution:** The dataset is slightly imbalanced (malignant < benign).  
- **Feature Correlation:** Strong correlations exist among size-related features (e.g., radius_mean, area_mean).  

### Example Plots:
- Histogram of features showing data spread.  
- Heatmap of feature correlations.  
- Bar chart for class distribution (benign vs malignant).  

These visualizations confirmed data readiness and guided preprocessing choices.  

---

##  Model Building
A **Neural Network classifier** was implemented using **TensorFlow/Keras**.  

### Model Architecture:
- **Input Layer:** 30 features.  
- **Hidden Layers:**  
  - Dense(32, activation='relu')  
  - Dense(16, activation='relu')  
- **Output Layer:** Dense(1, activation='sigmoid')  

### Optimizer & Loss:
- Optimizer: `Adam`  
- Loss Function: `Binary Crossentropy`  
- Metrics: `Accuracy`  

### Training:
- Epochs: 13  
- Batch Size: 32  
- Early stopping applied to avoid overfitting.  

---

##  Model Checking
- **Training Accuracy:** ~99%  
- **Validation Accuracy:** ~97%  
- **Test Accuracy:** ~96%  
- **Confusion Matrix:** Demonstrates high sensitivity (recall for malignant cases).  
- **Classification Report:**  
  - Precision, Recall, and F1-score for both classes.  

The model generalized well, showing consistent performance across all splits.  

---

##  Results & Insights
- The neural network successfully classified tumors with **~96% accuracy**.  
- Malignant tumors were detected with high recall, reducing the risk of false negatives (critical in medical use).  
- The project demonstrates how **AI can assist in medical diagnostics**, but human validation remains essential.  

---

##  Report & Documentation
This project followed a structured pipeline:  
1. **Problem Definition** – Early detection of breast cancer using ML.  
2. **Data Cleaning** – Preprocessing and feature scaling.  
3. **Analysis** – Visual exploration and feature correlation checks.  
4. **Model Building** – Neural Network with Keras.  
5. **Evaluation** – Accuracy, confusion matrix, and classification report.  
6. **Insights** – Strong model performance, potential clinical relevance.  

### Tools & Libraries:
- Python 3.8+  
- Numpy, Pandas, Matplotlib, Seaborn (Data Analysis & Visualization)  
- Scikit-learn (Dataset & Preprocessing)  
- TensorFlow/Keras (Model Development)  

---

##  How to Run
1. Clone this repository:
   ```bash
   git clone https://kajalkumari13/your-username/breast-cancer-nn.git
   cd breast-cancer-nn

