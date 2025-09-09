#  Breast Cancer Classification with Neural Network

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![Keras](https://img.shields.io/badge/Keras-Neural--Network-green)  

---

##  Overview
This project implements a **Neural Network model** to classify breast cancer tumors as **benign** (non-cancerous) or **malignant** (cancerous) using the **Breast Cancer Wisconsin Diagnostic dataset**.  
The model is designed to achieve **high accuracy (~95%)** while training efficiently (only **13 epochs** with EarlyStopping).  

The project demonstrates how **Artificial Intelligence** can be applied in the healthcare domain to assist doctors with **early detection and diagnosis**.  

---

##  Features
- End-to-end workflow: Data cleaning → preprocessing → modeling → evaluation.  
- Uses **Neural Networks with TensorFlow/Keras**.  
- Achieves **95% accuracy** in just **13 epochs**.  
- Generates **visualizations** (heatmap, histograms, class distribution).  
- Provides **confusion matrix and classification report** for evaluation.  
- Flexible and easy to extend (try new layers, optimizers, or other ML models).  

---

##  Requirements
- Python 3.8+  
- Libraries:  
  - numpy  
  - pandas  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - tensorflow
 
   
##  How the Code Works

- **Load Dataset:** Import breast cancer dataset from scikit-learn.  
- **Preprocess:** Scale features with StandardScaler.  
- **Split Data:** Train (70%), Validation (15%), Test (15%).  
- **Build Model:**  
  - Input layer (30 features).  
  - Two hidden layers (32 & 16 neurons, ReLU).  
  - Output layer (1 neuron, sigmoid).  
- **Train Model:**  
  - Optimizer: Adam.  
  - Loss: Binary Crossentropy.  
  - EarlyStopping → Stopped at 13 epochs.  
- **Evaluate:** Accuracy, confusion matrix, classification report.  
- **Visualize:** Plots of features, correlations, and training performance.  

---

## Building the Neural Network
The neural network used in this project:
![Neural Network](images/nn_structure.png)


---

##  Example Output

### Training Performance
- Training Accuracy: ~97%  
- Validation Accuracy: ~95%  
- Test Accuracy: ~95%  

### Confusion Matrix
|                   | Predicted Malignant | Predicted Benign |
|-------------------|----------------------|------------------|
| **Actual Malignant** | 59                   | 2                |
| **Actual Benign**    | 3                    | 106              |

### Classification Report
| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Malignant | 0.95      | 0.97   | 0.96     |
| Benign    | 0.96      | 0.94   | 0.95     |

---

## Training Visualization
Model Accuracy
Model Loss

---

##  Problem Definition & Understanding 

Breast cancer is a major health concern worldwide. Early detection can save lives.  
This project aims to create a predictive model that classifies tumors as **benign or malignant** using Neural Networks.

---

##  Data Collection & Cleaning 

- **Dataset:** Breast Cancer Wisconsin (Diagnostic).  
- **Features:** 30 numeric tumor cell features.  
- **Target:** Malignant (0) / Benign (1).  

**Preprocessing:**  
- Checked missing values (none).  
- Standardized features with StandardScaler.  
- Data split → Train (70%), Validation (15%), Test (15%).  

---

##  Data Analysis & Charts 

- **Class Distribution:** More benign than malignant.  
- **Correlation Heatmap:** Shows strong relationships between size-related features.  
- **Histograms & Boxplots:** Visualize distributions and outliers.  

---

##  Model Building 

**Neural Network Architecture:**  
- Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid).  

**Training Setup:**  
- Optimizer: Adam.  
- Loss: Binary Crossentropy.  
- Metrics: Accuracy.  
- Training stopped at **13 epochs (EarlyStopping).**  

---

##  Model Checking 

- **Test Accuracy:** 95%.  
- Confusion matrix confirms reliable predictions.  
- Classification report shows balanced precision/recall.  

---

##  Results & Insights 

- Neural Network achieved **95% accuracy**.  
- Very high recall for malignant tumors → reduces false negatives.  
- Proves that AI models can complement clinical diagnosis effectively.  

---

##  Report & Documentation 

**Workflow**  
1. Problem definition.  
2. Dataset preprocessing.  
3. Exploratory data analysis with charts.  
4. Neural Network design and training.  
5. Evaluation with metrics and confusion matrix.  
6. Insights and conclusion.  

**Tools**  
- Python  
- Scikit-learn  
- TensorFlow/Keras  
- Matplotlib  
- Seaborn  

---
## ⚙ Install All Dependencies

pip install -r requirements.txt
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

## Project Setup

1. Clone or download this repository.
2. Install dependencies:
    pip install -r requirements.txt
3.Run the Jupyter Notebook:
   jupyter notebook breast_cancer_nn.ipynb

---
   
##  Future Scope

- Hyperparameter tuning for better optimization.  
- Use advanced models (CNN, Ensemble Learning).  
- Add Explainable AI (LIME, SHAP) for interpretability.  
- Deploy as a web application for real-world use.  

---

##  Conclusion

This project demonstrates that a **Neural Network can classify breast cancer tumors with 95% accuracy in only 13 epochs.**  
The model is efficient, accurate, and interpretable with supporting charts and reports.  

AI solutions like this can support medical professionals in decision-making, leading to faster, more reliable cancer diagnosis.




