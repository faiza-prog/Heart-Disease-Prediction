import os
import json
import numpy as np
import pandas as pd
import joblib

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from django.conf import settings



@csrf_exempt
def predict_heart_disease(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Expect input like {"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}
            features = np.array(data['features']).reshape(1, -1)

            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, 'predictor', 'svm_model.pkl')

            model = joblib.load(model_path)

            prediction = model.predict(features)
            return JsonResponse({'prediction': int(prediction[0])})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)


@csrf_exempt
def train_model_view(request):
    if request.method == 'POST':
        try:
            csv_path = r'C:\django\heartpredictor\heart prediction disease project\heart.csv'
            df = pd.read_csv(csv_path)

            for column in df.columns:
                if df[column].dtype == 'object':
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])

            X = df.drop('HeartDisease', axis=1)
            y = df['HeartDisease']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = SVC()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
               
            model_path = os.path.join('C:\django\heartpredictor\predictor\svm_model.pkl')
            joblib.dump(model, model_path)

            return JsonResponse({'message': 'Model trained and saved successfully', 'accuracy': accuracy})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)


    
def predictor(request):
    return HttpResponse("This is the predictor endpoint.")

# Load your saved model
model_path = os.path.join(settings.BASE_DIR, 'C:\django\heartpredictor\predictor\svm_model.pkl')
model = joblib.load(model_path)

def show_form(request):
    if request.method == 'POST':
        try:
            # Get user input from the form
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

            # Create numpy array with same feature order as training
            input_features = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                                        fasting_bs, rest_ecg, max_hr, exercise_angina,
                                        oldpeak, st_slope]])

            # Make prediction
            prediction = model.predict(input_features)[0]

            # Map output to readable format
            result = "Yes" if prediction == 1 else "No"

            return render(request, 'predictor/result.html', {'result': result})

        except Exception as e:
            return render(request, 'predictor/form.html', {'error': str(e)})

    return render(request, 'predictor/form.html')