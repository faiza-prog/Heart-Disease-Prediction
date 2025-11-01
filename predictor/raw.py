# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def func_name():
    heart_df = pd.read_csv("C:\django\heartpredictor\heart prediction disease project\heart.csv")
    heart_df.head()
    new_heart_df = heart_df.copy()
    # EDA of heart_disease data 
    new_heart_df.shape
    new_heart_df.info()
    heart_df.drop(heart_df.iloc[:, 2:5], axis=1, inplace=True)
    new_heart_df.isnull().sum()
    # extract numerical columns
    num_col = new_heart_df.select_dtypes(exclude=["O"]).columns
    new_heart_df[num_col].head()
    # extract categorical columns
    cat_col = new_heart_df.select_dtypes(include=["O"]).columns
    new_heart_df[cat_col].head()
    def explore_data(df):
        print("DATA EXPLORATION")
        print(df.shape)
        print("STATISTICAL ANALYSIS OF NUMERIC DATA")
        print(df.describe().T)
    explore_data(new_heart_df)
    new_heart_df["ExerciseAngina"].unique()
    new_heart_df[cat_col].nunique()
    new_heart_df[num_col].nunique()
    new_heart_df.nunique()
    data = new_heart_df.groupby("ExerciseAngina")["Age"].count()
    data
    x_axis = list(data.index)
    y_axis = list(data.values)
    plt.bar(x_axis, y_axis)
    fig, axis = plt.subplots(3, 2, figsize=(15, 13))

    sns.countplot(x="ExerciseAngina", data=new_heart_df, ax=axis[0, 0])
    sns.countplot(x="ChestPainType", data=new_heart_df, ax=axis[0, 1])
    sns.countplot(x="RestingECG", data=new_heart_df, ax=axis[1, 0])
    sns.countplot(x="ST_Slope", data=new_heart_df, ax=axis[1, 1])
    sns.countplot(x="Sex", data=new_heart_df, ax=axis[2, 0])

    sns.boxplot(x="Sex", y="Age", data=new_heart_df)
    numerical_columns = new_heart_df[num_col].head()
    numerical_columns
    numerical_columns.corr()["HeartDisease"].sort_values(ascending=False)
    pd.get_dummies(new_heart_df["ChestPainType"], dtype=int)
    new_heart_df = pd.get_dummies(new_heart_df, drop_first=True, dtype=float)
    new_heart_df
    X = new_heart_df.drop("HeartDisease", axis=1).values
    y = new_heart_df["HeartDisease"].values
    new_heart_df.shape
    new_heart_df["HeartDisease"].unique()
    # count of each class
    new_heart_df["HeartDisease"].value_counts()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    np.random.seed(1)
    np.random.randint(1, 100, 10)
    scalar = StandardScaler()

    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    svm_model = SVC()

    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)

    accuracy_score(y_test, y_pred)

    svm_model.get_params()

    svm_model = SVC()
    # SVM

    param = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["linear", "poly", "rbf"],
    }

    # GridSearchCV
    grid_svc = GridSearchCV(estimator=svm_model, param_grid=param, cv=5)

    grid_svc.fit(X_train_scaled, y_train)  # training the model

    best_params = grid_svc.best_params_
    best_params
    best_estimator = grid_svc.best_estimator_
    best_estimator
    best_score = grid_svc.best_score_
    best_score
    final_svm_model = SVC(C=10, gamma=0.01, kernel="rbf")
    final_svm_model.fit(X_train_scaled, y_train)
    y_pred = final_svm_model.predict(X_test_scaled)
    accuracy_score(y_test, y_pred)
    



  # Save the final trained model
    joblib.dump(final_svm_model, 'svm_model.pkl')
    print("✅ Final SVM model saved as svm_model.pkl")
    
    joblib.dump(scalar, 'scalar.pkl')
    print("✅ Scalar saved as scaler.pkl")



#func_name()
