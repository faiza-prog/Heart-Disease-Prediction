# 🩺 Heart Disease Prediction Dashboard & System

## 📘 Overview
This project is a **Heart Disease Prediction System** developed as a **Final Year Project (FYP)**.  
This project predicts the likelihood of heart disease in patients using **Machine Learning**, provides a **web-based interface** built with **Django**, and visualizes key patterns using an **interactive Power BI dashboard**.

The system helps medical practitioners and users analyze heart health risk based on multiple medical attributes such as age, cholesterol level, blood pressure, and more.

The system not only predicts heart disease but also provides:
- **Heart ratio analysis**
- **Personalized health recommendations**
- **Interactive Power BI dashboard** for insights

---

## 🎯 Objectives
- Predict the presence of heart disease using machine learning models  
- Analyze patient medical data to identify risk factors  
- Calculate **heart ratio** for improved risk assessment  
- Provide **health recommendations** based on prediction results  
- Visualize data trends using an interactive dashboard  

---
## 🧠 Project Workflow

### 1️⃣ Data Preparation & Model Training (Python)
- Dataset: `heart.csv`
- Algorithms used:
  - **SVM (Support Vector Machine)** for classification
- Model training and evaluation done in `train_model.py` and `svc.ipynb`.
- Models are saved using **Joblib/Pickle** as:
  - `svm_model.pkl`
  - `scalar.pkl`

### 2️⃣ Web Application (Django)
- A simple web form built with **HTML + Django views** allows users to input their medical data.
- The trained ML model is loaded in Django’s backend to predict the presence of heart disease.
- Files include:
  - `views.py` – logic to handle prediction
  - `form.html` – user input page
  - `result.html` – displays prediction result (Heart Disease: Yes/No)
  - `urls.py`, `models.py`, `settings.py` – standard Django configuration
### 3️⃣ Data Visualization (Power BI)
- The Power BI dashboard visualizes:
  - Heart disease distribution by age and sex
  - Cholesterol and MaxHR trends
  - Chest pain type comparisons
- Dashboard name: **Heart Disease Prediction Dashboard**
- Example visuals:
  - Bar charts for age bins and sex ratio
  - Comparison of cholesterol and heart disease correlation

## Key Features
- ✅ Heart disease prediction (Yes / No)
- ✅ Heart ratio calculation
- ✅ Personalized health recommendations
- ✅ Machine learning–based decision making
- ✅ Interactive Power BI dashboard
- ✅ Easy-to-use interface
- ✅ Responsive web interface
---

## ⚙️ Machine Learning Models
- **Support Vector Machine (SVM)**
- **Linear Regression (Binary Classification)**

The models are trained on the **UCI Heart Disease Dataset** and saved using `joblib` for reuse.

---

## 📊 Dashboard Highlights

| Metric | Description | Value |
|------|------------|------|
| Patients with Heart Disease | Diagnosed individuals | **508** |
| Patients without Heart Disease | Healthy individuals | **410** |
| Total Records | Dataset size | **918** |

---

## 📈 Visual Insights
- **Age Distribution:** Highest heart disease cases between 50–60 years  
- **MaxHR Analysis:** Healthy individuals show higher MaxHR  
- **Chest Pain Types:** ASY type is most common in heart disease patients  
- **Gender Analysis:** Males have higher heart disease occurrence  
- **Cholesterol Trends:** Slight variations between healthy and affected patients  

---

## 🩸 Heart Ratio & Recommendations
- Heart ratio is calculated to enhance prediction reliability  
- Based on results, the system provides:
  - Lifestyle improvement suggestions
  - Diet and exercise recommendations
  - Health awareness guidance

---

## ⚙️ Filters (Dashboard)
- Chest Pain Type  
- Gender  
- ST_Slope  
- Age  

These filters allow dynamic exploration of patient data.

---

## 🧰 Tools & Technologies

- **Backend**: Python, Django
- **Machine Learning**: Scikit-learn (SVM), Pandas, NumPy, Joblib
- **Frontend**: HTML, CSS, Bootstrap (Django templates)
- **Data Visualization**: Power BI
- **Dataset**: UCI Heart Disease Dataset

---
## ⚙️ How to Run the Project

### Step 1: Clone this repository
```bash
git clone https://github.com/faiza-prog/heart-disease-prediction.git
cd heart-disease-prediction
```

### Step 2: Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # for Windows
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Django web server
```bash
python manage.py runserver
```
Open your browser and go to **http://127.0.0.1:8000/**  

### Step 5: View the Power BI Dashboard
Open the `.pbix` file in Power BI Desktop or view exported images (in the `/dashboard` folder if uploaded).

---

## 📊 Model Performance
| Algorithm | Accuracy | Description |
|------------|-----------|--------------|
| SVM        | ~85–90%   | Best performing classifier |

## 🩺 Key Insights
- Age and MaxHR are strong indicators of heart disease  
- Male patients show higher risk in the dataset  
- Certain chest pain types strongly correlate with heart disease  

---

## 🎓 Academic Context
This project is developed as a **Final Year Project (FYP)** for the **Computer Science** program.

---

## 📌 Future Enhancements
Add more ML models for comparison 
Improve recommendation system  
Deploy system online  
Add real-time data input  
---

## 👩‍💻 Author
**Faiza Batool**  
Final Year Student – Computer Science  
The Islamia University Of Bahawalpur
Department of Computer Science
