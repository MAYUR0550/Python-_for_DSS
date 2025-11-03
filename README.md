---

# 🧠 Data Science with Python (DSS Project)

## 📘 Overview

This project demonstrates key **data science and machine learning** concepts using Python.
We analyze real-world datasets from Kaggle and perform data analysis, visualization, and predictive modeling using various algorithms.

---

## 📂 Datasets Used

| Dataset Name      | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `titanic.csv`     | Passenger survival prediction dataset (classification)          |
| `diabetes.csv`    | Medical dataset predicting diabetes occurrence (classification) |
| `salary_data.csv` | Simple salary prediction dataset (regression)                   |
| `framingham.csv`  | Heart disease dataset for risk prediction (classification)      |

---

## 🧾 Contents

### 1️⃣ Data Acquisition using Pandas

* Imported CSV files using `pandas.read_csv()`
* Explored datasets with `.head()`, `.info()`, `.describe()`
* Checked dataset dimensions and structure

### 2️⃣ Central Tendency Measures

* Computed **Mean**, **Median**, and **Mode** using:

  ```python
  df['column'].mean()
  df['column'].median()
  df['column'].mode()
  ```

### 3️⃣ Basics of DataFrame

* Created and manipulated DataFrames
* Accessed rows and columns
* Used indexing, slicing, and filtering operations

### 4️⃣ Missing Values Treatment

* Identified missing values using `.isnull().sum()`
* Treated missing data by:

  * Dropping null values with `dropna()`
  * Imputing using `fillna()` with mean/median/mode

### 5️⃣ Creation of Arrays using NumPy

* Created 1D, 2D arrays using:

  ```python
  np.array(), np.arange(), np.linspace(), np.random.rand()
  ```
* Performed mathematical operations and reshaping

### 6️⃣ Data Visualization

* Visualized data using:

  * **Matplotlib** and **Seaborn**
  * Common plots: histogram, boxplot, scatterplot, heatmap

  ```python
  sns.heatmap(df.corr(), annot=True)
  plt.scatter(x, y)
  ```

### 7️⃣ Simple Linear Regression

* Built a model using `sklearn.linear_model.LinearRegression`
* Trained model on `salary_data.csv`
* Evaluated performance using R² and MSE

### 8️⃣ Logistic Regression

* Implemented using `LogisticRegression` from sklearn
* Used `titanic.csv` and `diabetes.csv`
* Evaluated using confusion matrix, accuracy, precision, recall

### 9️⃣ K-Nearest Neighbors (KNN)

* Used `KNeighborsClassifier`
* Tuned `k` value using accuracy comparison
* Applied on classification datasets

### 🔟 Support Vector Machine (SVM)

* Implemented using `SVC`
* Tried linear, polynomial, and RBF kernels
* Compared accuracy across kernels

### 11 🌳 Decision Tree

* Implemented Decision Tree model using DecisionTreeClassifier and DecisionTreeRegressor
* Applied for both classification and regression tasks
* Visualized tree structure and interpreted decision paths
* Evaluated performance using accuracy and feature importance

### 12🌲 Random Forest Classifier

* Implemented Random Forest using RandomForestClassifier
* Utilized ensemble learning by combining multiple decision trees
* Improved model stability and reduced overfitting
* Evaluated using accuracy, precision, recall, and feature importance

---

## 🧰 Technologies Used

* **Python 3.x**
* **Libraries:**

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`

---

## 📊 Results & Insights

* Learned data preprocessing and handling missing data
* Visualized key statistical relationships
* Built regression and classification models
* Compared performance of ML algorithms

---

## 🚀 Future Scope

* Add hyperparameter tuning using `GridSearchCV`
* Implement feature selection and scaling
* Deploy models using Streamlit or Flask

---

## 👨‍💻 Author

**Mayur Waghmare**
*Data Science Student | Python Enthusiast*

---

