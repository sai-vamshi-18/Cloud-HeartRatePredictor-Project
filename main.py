import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
cat_encoders = pickle.load(open('cat_encoders.pkl','rb'))
cont_encoder = pickle.load(open('cont_encoder.pkl','rb'))
columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia']
cat_features = ["sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg", "exercise_induced_angina", "st_slope", "num_major_vessels", "thalassemia"]
cont_features = ["age", "resting_blood_pressure", "cholesterol", "max_heart_rate_achieved", "st_depression"]

@app.route('/')
def home():
    return render_template('home.html', prediction_text="")

@app.route('/predict',methods = ['POST'])
def predict():
    form_input = request.form.values()
    form_input = list(form_input)
    print(form_input)
    for i in [0, 3, 4, 7, 9]:
        form_input[i] = float(form_input[i])

    final_features = pd.DataFrame([form_input], columns=columns)
    final_features = preprocess_df(final_features, cat_features, cont_features)
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        return render_template('home.html', prediction_text="You don't have heart disease")
    else:
        return render_template('home.html', prediction_text="You have a heart disease, Please consult doctor")


def preprocess_df(df, cat_features, cont_features):
    for col in cat_features:
        df[col] = cat_encoders[col].transform(df[col])
     
    df[cont_features] = cont_encoder.transform(df[cont_features])
    return df

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)