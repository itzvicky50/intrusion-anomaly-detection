# Intrusion and Anomaly Detection in Smart Home Ecosystem

This repository contains the implementation of machine learning models for intrusion and anomaly detection in smart home environments. The project applies classical machine learning algorithms to detect intrusions and anomalies in smart home environments, utilizing preprocessing techniques like SMOTE to mitigate class imbalance and enhance model performance.

---

## 📚 Project Overview

This study focuses on detecting network intrusions and anomalies in smart home ecosystems using supervised machine learning algorithms, including:

- K-Nearest Neighbour (KNN)  
- Random Forest (RF)  
- Decision Tree (DT)  
- Logistic Regression (LR)  
- Support Vector Classifier (SVC)  

The models were trained and evaluated using the **CIC IoT 2023 Smart Home Dataset**, employing evaluation metrics such as accuracy, precision, recall, F1-score, Confusion Matrix and ROC-AUC.

---

## 📁 Project Structure

```
📂 intrusion-anomaly-detection
├── .ipynb_checkpoints/             # Jupyter Notebook Checkpoints
├── intrusion_anomaly_detection.html # HTML Export of Jupyter Notebook (Results)
├── intrusion_anomaly_detection.ipynb # Main Jupyter Notebook
└── smarthome_data.csv              # Dataset File
```

---

## 🗂 Dataset Access

This project uses the **CIC IoT 2023 Smart Home Dataset** for training and evaluation.  

**Dataset Location in Project (please extract it to get the data in csv):**  
- `smarthome_data.zip`  

The dataset used in this project was obtained specifically from the merged CSV files available on the official CIC repository.  

- **Official Dataset Information Page:**  
  [Visit Here](https://www.unb.ca/cic/datasets/iotdataset-2023.html)  

- **Direct Dataset Download Link:**  
  [Download Merged CSV Files](http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/MERGED_CSV/)  

> *Note: The merged CSV files (e.g., Merged01.csv, Merged02.csv) were used to create the `smarthome_data.csv` used in this project.*

---

## ⚙️ Experimental Environment

- **Operating System**: Windows 11 Pro (Build 26100)  
- **Processor**: Intel® Core™ i5-8350U CPU @ 1.70GHz  
- **RAM**: 16 GB  
- **GPU**: Intel® UHD Graphics 620  
- **Python Version**: 3.10.8  
- **Development Tools**: Jupyter Notebook  

### 📦 Key Libraries  
- pandas: 2.2.2  
- numpy: 1.26.4  
- matplotlib: 3.9.2  
- seaborn: 0.13.2  
- scikit-learn: 1.5.1  
- imbalanced-learn: 0.12.3  

---

## ▶️ Usage Instructions

1. **Clone the Repository:**  
```bash
git clone https://github.com/itzvicky50/intrusion-anomaly-detection.git
cd intrusion-anomaly-detection
```

2. **Ensure the Dataset (`smarthome_data.csv`) is placed in the project root directory.**

3. **Run the Experiment:**  
```bash
jupyter notebook intrusion_anomaly_detection.ipynb
```

4. **View Results:**  
- Results can be explored directly in the Jupyter Notebook or by opening `intrusion_anomaly_detection.html` in a web browser.

---

## 📢 Disclaimer

This repository is intended for academic and research purposes only. Please ensure compliance with the CIC dataset licensing terms when using the dataset.
