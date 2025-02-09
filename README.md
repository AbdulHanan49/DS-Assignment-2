# Life Expectancy Prediction - Data Analysis & Machine Learning

## Overview
This project focuses on analyzing a dataset containing various socioeconomic and health-related factors to predict life expectancy. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model training using Random Forest, XGBoost, and Gradient Boosting Regressor.

## Dataset
The dataset is stored in Google Drive and is loaded using pandas. It contains various economic, health, and infrastructure-related indicators that can influence life expectancy.

## Steps Performed
### 1. **Data Loading & Preprocessing**
- Mounted Google Drive and loaded the dataset.
- Removed unnamed columns and transposed data for better analysis.
- Checked for missing values and handled them by replacing with the mean or encoding categorical variables.
- Standardized numerical features for improved model performance.

### 2. **Exploratory Data Analysis (EDA)**
- Displayed dataset structure and statistical summary.
- Visualized missing values using `missingno`.
- Checked unique values in categorical columns.
- Plotted joint plots to examine relationships between variables.
- Computed a correlation heatmap to identify important relationships.

### 3. **Feature Engineering**
- Converted categorical features using Label Encoding.
- Transformed values (e.g., `per 100` converted to `per 1000` in `internet_users`).
- Selected relevant features based on correlation analysis.

### 4. **Data Splitting & Standardization**
- Splitted data into training (80%) and testing (20%) sets.
- Standardized numerical features except key categorical variables.

### 5. **Machine Learning Models**
#### **Random Forest Regressor**
- Trained using 200 estimators.
- Evaluated using Mean Absolute Error (MAE).

#### **XGBoost Regressor**
- Trained with 200 estimators.
- Measured MAE and R-squared score.

#### **Gradient Boosting Regressor**
- Trained with 200 estimators.
- Compared MAE and R-squared score with other models.

## Results
- The performance of each model was evaluated using MAE and R-squared score.
- Feature selection and engineering significantly impacted model accuracy.
- Gradient Boosting and XGBoost achieved the best performance in predicting life expectancy.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
  
To run the project, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn missingno scikit-learn xgboost
```

## How to Run
1. Clone the repository.
2. Install required dependencies.
3. Load the dataset in the specified path.
4. Run the script to train and evaluate models.

## Future Improvements
- Tune hyperparameters for better model performance.
- Use advanced feature selection techniques.
- Explore deep learning models for better accuracy.

## Recommendations
If you have any recommendations or suggestions, please feel free to share them with us. 
We appreciate your feedback!
