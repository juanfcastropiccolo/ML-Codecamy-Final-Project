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
   - **Original Approach**: Random Forest Regressor with hyperparameter tuning using **GridSearchCV** and validation folds.
   - **Improved Approach**: Transitioned to `RandomForestLearner` from **YDF** (Google's TensorFlow Decision Forests) for better efficiency and compatibility with large datasets.

---

## **Updates from Today's Work**

### **Key Improvements**:
1. **Optimized Hyperparameters**:
   - After extensive hyperparameter tuning, we determined the following best parameters for the Random Forest Learner:
     ```python
     {
         'num_trees': 50,
         'max_depth': 20,
         'min_examples': 2
     }
     ```

2. **Avoided Redundant Training**:
   - By leveraging these hyperparameters directly, we skipped retraining for less promising combinations, significantly reducing computation time.

3. **Transition to YDF**:
   - Replaced scikit-learn's Random Forest implementation with `ydf.RandomForestLearner` for compatibility with large datasets and efficient tree-based modeling.

4. **Improved Pipeline**:
   - Modified the ML pipeline to include PCA and scaling while adapting the training process to work seamlessly with YDF.

---

## **Results**
- **Previous Model**: 
  - **Simple Linear Regression**: \( R^2 = 0.6079 \)
  - **Random Forest Regressor**: \( R^2 = 0.9999 \) (scikit-learn implementation).

- **Current Model**:
  - **RandomForestLearner**: \( R^2 = 0.99996 \), achieving near-perfect predictions with reduced training time.

### **Visualizations**:
1. Predicted vs. Actual Values:
   - Visualized the relationship between predicted and actual wave heights, showing a strong correlation.
2. Feature Importance:
   - The model identified significant wave height and wind wave characteristics as key predictors.

---

## **Future Work**
- Integrate GPU acceleration for larger datasets.
- Experiment with other tree-based algorithms such as Gradient Boosted Trees (GBT) in YDF.
- Automate hyperparameter tuning using Bayesian Optimization or similar techniques.
