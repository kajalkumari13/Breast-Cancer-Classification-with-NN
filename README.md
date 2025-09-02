#  Breast Cancer Classification with Neural Network

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

##  Overview
This project is a **Data Science & Machine Learning application** that classifies breast cancer tumors as **benign (non-cancerous)** or **malignant (cancerous)** using a **Neural Network (NN)** built with TensorFlow/Keras.  
The goal is to assist in **early cancer detection** through predictive modeling.  

---

##  Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Samples:** 569  
- **Features:** 30 numeric features (radius, texture, smoothness, etc.)  
- **Target Classes:**  
  - `0 = Malignant`  
  - `1 = Benign`  

---

##  Requirements
The project requires the following dependencies:

\`\`\`
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
joblib
\`\`\`

Install them with:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

##  Repository Structure
\`\`\`
breast-cancer-classification/
│── breast_cancer_nn.py        # Main script for training and evaluation
│── requirements.txt           # Required dependencies
│── README.md                  # Documentation
│── results/                   # Stores generated plots
│    ├── confusion_matrix.png
│    ├── training_curves.png
│    └── roc_curve.png
│── models/                    # Stores trained model and scaler
│    ├── breast_cancer_nn.h5
│    └── scaler.joblib
\`\`\`

---

##  Installation & Setup
1. Clone this repository:
   \`\`\`bash
   git clone https://github.com/kajalkumari13/breast-cancer-classification.git
   cd breast-cancer-classification
   \`\`\`

2. Create a virtual environment (recommended):
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   \`\`\`

3. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

---

##  How to Run
Run the main script to train and evaluate the Neural Network:

\`\`\`bash
python breast_cancer_nn.py
\`\`\`

This will:
1. Train the Neural Network model  
2. Evaluate accuracy, precision, recall, F1-score, and ROC-AUC  
3. Save plots (Confusion Matrix, Training Curves, ROC Curve) inside \`results/\`  
4. Save the trained model and scaler inside \`models/\`  

---

##  Results
- The Neural Network achieves **95–98% accuracy** on test data.  

### Example Outputs:
**Confusion Matrix**  
![Confusion Matrix](results/confusion_matrix.png)

**Training Curves**  
![Training Curves](results/training_curves.png)

**ROC Curve**  
![ROC Curve](results/roc_curve.png)

---

##  Future Scope
- Compare with ML algorithms like **SVM, Random Forest, XGBoost**  
- Perform **Hyperparameter Tuning** (GridSearchCV, Optuna)  
- Deploy as a **Flask/Streamlit web app** for real-time predictions  
- Extend methodology to other **medical datasets**  

---

##  Acknowledgements
- Dataset: [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- Libraries: scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  
