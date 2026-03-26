# Parkinson's Disease Detection via Voice Analysis 🎤🔬

[[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/parkinsons-voice-detection/blob/main/Parkinsons.ipynb)
[[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **A comprehensive machine learning approach to screening Parkinson's Disease through voice biomarker analysis, achieving up to 95% accuracy.**

## 🌟 Overview

This project demonstrates how machine learning can enable **telediagnosis** of Parkinson's Disease (PD) through voice pattern analysis. By analyzing acoustic features extracted from speech samples, we can identify biomarkers associated with:

- **Dysphonia** (defective voice use)
- **Hypophonia** (reduced volume)
- **Monotone** (reduced pitch range)
- **Dysarthria** (articulation difficulties)

### Key Results

| Model | Accuracy | Use Case |
|-------|----------|----------|
| **XGBoost Classifier** | 98% | Recommended for production |
| **K-Nearest Neighbors** | 98% | Fast screening tool |
| **Decision Trees** | 96% | Interpretable results |
| **Support Vector Machine** | 90% | Robust to outliers |
| **Logistic Regression** | 88% | Baseline comparison |
| **Gaussian Naive Bayes** | 69% | Probabilistic approach |

## 📊 Dataset

The dataset contains **195 voice recordings** from:
- **147 Parkinson's patients** (6 female, 141 male)
- **48 healthy controls** (10 female, 38 male)

### Voice Features Extracted (22 total)

#### Frequency-based Features
- `MDVP:Fo(Hz)` - Average vocal fundamental frequency
- `MDVP:Fhi(Hz)` - Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)` - Minimum vocal fundamental frequency

#### Jitter Measures (variation in frequency)
- `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`
- `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`

#### Shimmer Measures (variation in amplitude)
- `MDVP:Shimmer`, `MDVP:Shimmer(dB)`
- `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`

#### Noise-to-Harmonics Ratios
- `NHR`, `HNR`

#### Nonlinear Dynamics Features
- `RPDE`, `DFA` - Complexity measures
- `spread1`, `spread2` - Signal variability
- `D2` - Correlation dimension
- `PPE` - Pitch period entropy

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Run the first code cell to upload required files
3. Upload `parkinsons.csv` and `van_plot.py` when prompted
4. Run all cells sequentially

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/parkinsons-voice-detection.git
cd parkinsons-voice-detection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Parkinsons.ipynb
```

### Required Files

- `Parkinsons.ipynb` - Main analysis notebook
- `parkinsons.csv` - Voice feature dataset
- `van_plot.py` - Custom visualization library

## 📋 Requirements

```
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.3.0
scipy>=1.6.0
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Normalization**: MinMax scaling to [-1, 1] interval
- **Train/Test Split**: 75% training, 25% testing
- **Stratified Sampling**: Maintains class distribution

### 2. Model Training & Evaluation

We evaluate six classification algorithms using:

#### Performance Metrics

**Accuracy**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Sensitivity (Recall)**
$$\text{Sensitivity} = \frac{TP}{TP + FN}$$

**Specificity**
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Matthews Correlation Coefficient**
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

Where: TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

### 3. Visualization

Each model generates:
- **Confusion Matrix** - Classification breakdown
- **ROC Curve** - True vs. False positive rates
- **AUC Score** - Area under ROC curve
- **Classification Report** - Precision, recall, F1-score

## 📈 Sample Results

### XGBoost Classifier Performance

```
Model Accuracy: 98.0%

              precision    recall  f1-score   support

           0       0.92      1.00      0.96        12
           1       1.00      0.97      0.99        37

    accuracy                           0.98        49
   macro avg       0.96      0.99      0.97        49
weighted avg       0.98      0.98      0.98        49
```

### Visual Outputs

The notebook generates:
- Feature distribution histograms (7 plots)
- Confusion matrices for each model
- ROC curves with AUC scores
- Comparative performance tables

## 🎯 Clinical Applications

### Screening Tool
- **Remote Assessment**: Voice samples collected via phone/telemedicine
- **Early Detection**: Identify subtle vocal changes before motor symptoms
- **Longitudinal Monitoring**: Track disease progression over time

### Research Applications
- **Biomarker Discovery**: Identify new vocal signatures of PD
- **Treatment Efficacy**: Measure therapeutic response through voice changes
- **Phenotype Classification**: Distinguish PD subtypes via vocal patterns

### Limitations
- Dataset imbalance (75% positive, 25% negative)
- Limited demographic diversity in original cohort
- Requires quality audio recordings
- Not a replacement for clinical diagnosis

## 🛠️ Advanced Usage

### Custom Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Define custom model
custom_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train and evaluate
fit_predict_summarize(custom_model, 
                     'Random Forest', 
                     X_train, Y_train, 
                     X_test, Y_test)
```

### Feature Importance Analysis

```python
# Get feature importances from XGBoost
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, Y_train)

# Plot feature importance
xgb.plot_importance(model)
plt.show()
```

## 📚 Scientific Background

### Original Research

This work builds on the foundational paper:
> **"Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings"**

The original study used **Praat acoustic analysis software** to extract time-frequency features from parsed speech samples (specific utterances like vowel 'o' and word 'four').

### Key Insights

1. **Frequency Domain Analysis**: Parkinson's affects vocal cord control, creating distinctive frequency patterns
2. **Jitter & Shimmer**: Increased variation in pitch (jitter) and amplitude (shimmer) are hallmarks of PD
3. **Nonlinear Dynamics**: Complexity measures capture subtle irregularities in vocal fold vibration
4. **High-Frequency Content**: PD patients show elevated high-frequency components in speech

## 🤝 Contributing

We welcome contributions from:
- **Clinicians**: Validation studies, clinical feedback
- **Data Scientists**: Model improvements, feature engineering
- **Researchers**: Dataset expansion, cross-validation studies

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 References

- [Original Parkinson Speech Paper](http://bit.ly/3bFg455)
- [Praat Acoustic Analysis Software](https://www.fon.hum.uva.nl/praat/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 👥 Credits

**Current Implementation**
- L. Van Warren - Analysis and documentation

**Original Research & Code**
- [Shlok Khandelwal](https://github.com/shlokKh) - Initial codebase
- [Elcin Ergin, Shu Hayakawa, Timardeep Kaur](https://github.com/hayakshu/Classification-Analysis-Of-Parkinson-Speech-Dataset) - Classification analysis
- [Dennis T](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea) - Confusion matrix visualization

**Academic Supervision**
- Prof. Mariofanna Milanova - University of Arkansas, Little Rock
- TA. Imran 'Md' Sarker

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This tool is for research and educational purposes only. It is NOT a medical diagnostic device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.**

Always consult qualified healthcare providers for:
- Parkinson's Disease diagnosis
- Treatment recommendations
- Medical decision-making

## 📧 Contact

For questions, collaborations, or clinical validation opportunities:
- **Project Issues**: [GitHub Issues](https://github.com/yourusername/parkinsons-voice-detection/issues)
- **Email**: your.email@example.com

---

<div align="center">

**If this project helps your research or clinical practice, please consider citing it and starring the repository! ⭐**

Made with ❤️ for advancing Parkinson's Disease research

</div>
