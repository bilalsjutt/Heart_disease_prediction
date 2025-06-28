# Heart Disease Prediction Using Machine Learning

A comprehensive machine learning pipeline for predicting heart disease using multiple classification algorithms with proper data preprocessing, cross-validation, and model comparison.

## 🚀 Overview

This project implements a robust machine learning pipeline to predict heart disease status using various patient health indicators. The system compares multiple classification algorithms and selects the best-performing model based on rigorous evaluation metrics.

## 🎯 Features

- **Multiple Model Comparison**: Evaluates 5 different classification algorithms
- **Proper Data Handling**: Implements data leakage prevention with correct train-test splitting
- **Advanced Preprocessing**: Handles both numerical and categorical features appropriately
- **Class Balancing**: Uses SMOTE (Synthetic Minority Oversampling Technique) for imbalanced datasets
- **Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **Comprehensive Metrics**: Detailed performance analysis with multiple evaluation metrics

## 📋 Requirements

```bash
pandas
numpy
scikit-learn
imbalanced-learn
warnings
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## 📊 Dataset Requirements

The code expects a CSV file with the following characteristics:
- Target column: `"Heart Disease Status"` with values `"No"` and `"Yes"`
- Mixed data types: numerical and categorical features
- Potential ordinal features: `'Exercise Habits'`, `'Stress Level'`, `'Sugar Consumption'`

### Expected Data Structure
```
- Numerical columns: Age, blood pressure, cholesterol levels, etc.
- Categorical columns: Gender, smoking status, family history, etc.
- Ordinal columns: Exercise habits (Low/Medium/High), Stress level, Sugar consumption
- Target: Heart Disease Status (No/Yes)
```

## 🔧 Usage

1. **Prepare your dataset**: Ensure your CSV file matches the expected format
2. **Update file path**: Modify the dataset path in the code:
   ```python
   df = pd.read_csv("your_heart_dataset.csv")  # Replace with actual file path
   ```
3. **Run the script**:
   ```bash
   python heart_disease_prediction.py
   ```

## 🤖 Models Evaluated

The pipeline evaluates the following algorithms:

1. **Random Forest Classifier**
   - n_estimators=100, max_depth=8
   - Good for feature importance analysis

2. **Gradient Boosting Classifier**
   - n_estimators=100, max_depth=6
   - Excellent for complex patterns

3. **Logistic Regression**
   - max_iter=1000
   - Interpretable linear model

4. **Support Vector Machine (SVM)**
   - RBF kernel with probability estimation
   - Effective for high-dimensional data

5. **K-Nearest Neighbors (KNN)**
   - n_neighbors=5
   - Simple, instance-based learning

## 🔄 Pipeline Architecture

### 1. Data Preprocessing
- **Numerical Features**: Mean imputation + Standard scaling
- **Ordinal Features**: Most frequent imputation + Ordinal encoding
- **Nominal Features**: Most frequent imputation + One-hot encoding

### 2. Class Balancing
- SMOTE implementation for handling imbalanced datasets
- Applied after preprocessing to prevent data leakage

### 3. Model Training & Evaluation
- 5-fold stratified cross-validation
- Final evaluation on hold-out test set (20% of data)
- Comprehensive metrics calculation

## 📈 Output

The script provides:

### Cross-Validation Results
- Individual CV scores for each model
- Mean CV score with standard deviation
- Model ranking based on CV performance

### Test Set Evaluation
- Final accuracy on unseen test data
- Best model selection
- Detailed classification report
- Confusion matrix analysis

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Feature importance (when available)
- Comparison table of all models

### Sample Output
```
BEST MODEL: Random Forest
===============================================
Cross-Validation Score: 0.8542
Test Set Accuracy: 0.8421

Detailed Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.83      0.84       95
           1       0.84      0.86      0.85       105
```

## 🛡️ Best Practices Implemented

- ✅ **Data Leakage Prevention**: Train-test split before any preprocessing
- ✅ **Proper Cross-Validation**: Stratified K-fold for imbalanced data
- ✅ **Pipeline Architecture**: All preprocessing steps in scikit-learn pipelines
- ✅ **Class Balancing**: SMOTE applied within pipeline
- ✅ **Feature Scaling**: StandardScaler for numerical features
- ✅ **Robust Evaluation**: Multiple metrics and model comparison

## 📝 Customization

### Adding New Models
```python
models['New Model'] = YourClassifier(parameters)
```

### Modifying Preprocessing
Update the respective pipelines in the `ColumnTransformer`:
```python
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Change strategy
    ('scaler', RobustScaler())  # Use different scaler
])
```

### Adjusting Cross-Validation
```python
cv_folds = 10  # Increase folds
cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new classification algorithms
- Implementing advanced feature engineering
- Adding hyperparameter tuning capabilities
- Improving visualization of results

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔍 Future Enhancements

- [ ] Automated hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Feature selection techniques
- [ ] Advanced ensemble methods
- [ ] Model interpretability tools (SHAP, LIME)
- [ ] Web interface for predictions
- [ ] Model persistence and loading capabilities

---

**Note**: Ensure your dataset follows the expected format and update the file path before running the script. The pipeline is designed to be robust and handle various data quality issues automatically.
