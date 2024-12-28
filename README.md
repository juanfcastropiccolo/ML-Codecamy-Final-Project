# ML-Codecamy-Final-Project
**Codecademy's Machine Learning Career Path - Final Project (End-to-end ML Pipeline)**

---

## Project Scope

### **Goals**:
- Predict the height of ocean waves using machine learning techniques.

### **Dataset**:
- **Source**: [Global Ocean Waves Analysis and Forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_WAV_001_027/download?dataset=cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411)
- **Variable of Interest**: Sea surface wave maximum height (**VCMX**) in meters.

### **Analysis**:
- Build an end-to-end ML pipeline to predict the height of waves based on oceanographic and meteorological features.

---

## **Pipeline Overview**
1. **Preprocessing**:
   - Features extracted from the dataset: Latitude, longitude, significant wave height, swell characteristics, wind wave characteristics, etc.
   - Handled missing values and standardized the data.

2. **Dimensionality Reduction**:
   - Applied **PCA** to reduce the number of features while retaining variability.

3. **Model**:
   - Random Forest Regressor.
   - Hyperparameter tuning using **GridSearchCV** with validation folds.

---

## **Results**
- **Best Model**: Random Forest Regressor with the following hyperparameters:
  
  ```python
  {
      'bootstrap': True,
      'max_depth': 20,
      'max_features': 'sqrt',
      'min_samples_leaf': 1,
      'min_samples_split': 5,
      'n_estimators': 100
  }
