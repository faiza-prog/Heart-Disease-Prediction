import numpy as np
import joblib
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from django.http import JsonResponse

# ===========================
# Form-based prediction
# ===========================
def show_form(request):
    result = None
    risk_percent = None
    heart_efficiency = None
    risk_level = None
    exercise = []

    if request.method == 'POST':
        try:
            age = int(request.POST.get('Age'))
            sex = int(request.POST.get('Sex'))
            chest_pain = int(request.POST.get('ChestPainType'))
            resting_bp = int(request.POST.get('RestingBP'))
            cholesterol = int(request.POST.get('Cholesterol'))
            fasting_bs = int(request.POST.get('FastingBS'))
            rest_ecg = int(request.POST.get('RestingECG'))
            max_hr = int(request.POST.get('MaxHR'))
            exercise_angina = int(request.POST.get('ExerciseAngina'))
            oldpeak = float(request.POST.get('Oldpeak'))
            st_slope = int(request.POST.get('ST_Slope'))

            input_features = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                                        fasting_bs, rest_ecg, max_hr, exercise_angina,
                                        oldpeak, st_slope]])

            # Load model and scaler
            model = joblib.load(r'C:\django\heartpredictor\predictor\svm_model.pkl')
            scaler = joblib.load(r'C:\django\heartpredictor\predictor\scaler.pkl')

            # Scale features
            input_scaled = scaler.transform(input_features)

            # Predict
            prediction = model.predict(input_scaled)[0]
            result = "Yes" if prediction == 1 else "No"

            # Risk probability
            proba = model.predict_proba(input_scaled)[0][1] * 100
            risk_percent = round(proba, 2)

        
            

            # Risk level & exercise
            if risk_percent <= 10:
                risk_level = "Very Low Risk"
                exercise = ["Light walking (10-15 min)", "Gentle stretching", "Deep breathing"]
            elif risk_percent <= 30:
                risk_level = "Low Risk"
                exercise = ["Brisk walking (20-30 min)", "Yoga", "Low-impact cardio"]
            elif risk_percent <= 60:
                risk_level = "Moderate Risk"
                exercise = ["Moderate cardio (20-30 min)", "Strength training 2-3/week", "Swimming or jogging"]
            else:
                risk_level = "High Risk"
                exercise = ["Consult a doctor before exercise", "Light supervised activity", "Focus on diet"]

        except Exception as e:
            return render(request, 'predictor/form.html', {'error': str(e)})

    return render(request, 'predictor/form.html', {
        'result': result,
        'risk_percent': risk_percent,
        'heart_efficiency': heart_efficiency,
        'risk_level': risk_level,
        'exercise': exercise
    })


# ===========================
# Train model API
# ===========================
@csrf_exempt
def train_model_view(request):
    if request.method == 'POST':
        try:
            df = pd.read_csv(r'C:\django\heartpredictor\heart prediction disease project\heart.csv')

            for column in df.columns:
                if df[column].dtype == 'object':
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])

            X = df.drop('HeartDisease', axis=1)
            y = df['HeartDisease']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(probability=True)
            model.fit(X_train_scaled, y_train)

            accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

            joblib.dump(model, r'C:\django\heartpredictor\predictor\svm_model.pkl')
            joblib.dump(scaler, r'C:\django\heartpredictor\predictor\scaler.pkl')

            return JsonResponse({'message': 'Model trained and saved successfully', 'accuracy': accuracy})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)


# ===========================
# JSON API prediction
# ===========================
@csrf_exempt
def predict_heart_disease(request):
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            features = np.array(data['features']).reshape(1, -1)

            model = joblib.load(r'C:\django\heartpredictor\predictor\svm_model.pkl')
            scaler = joblib.load(r'C:\django\heartpredictor\predictor\scaler.pkl')

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1] * 100

            return JsonResponse({'prediction': int(prediction), 'risk_percent': round(probability, 2)})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
