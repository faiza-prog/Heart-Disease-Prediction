import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('C:\django\heartpredictor\predictor\svm_model.pkl')

# Sample input (raw, human-readable)
sample_input = {
    'Age': 55,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 140,
    'Cholesterol': 289,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 172,
    'ExerciseAngina': 'N',
    'Oldpeak': 1.5,
    'ST_Slope': 'Up'
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_input])

# Load same dataset used during training
df = pd.read_csv(r'C:\django\heartpredictor\heart prediction disease project\heart.csv')

# Encode the test input using same LabelEncoders
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(df[column])  # fit on original data
        sample_df[column] = le.transform(sample_df[column])

# Make prediction
prediction = model.predict(sample_df)

# Show result
if prediction[0] == 1:
    print(" Prediction: Patient has Heart Disease")
else:
    print(" Prediction: Patient does NOT have Heart Disease")
