from flask import session
import os
from database import get_db
from auth import signup_user, login_user
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect
import numpy as np
import joblib
import shap
import pandas as pd

# chatbot import AFTER env loaded
from chatbot import get_chat_response

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")

@app.route('/heart')
def heart():
    return render_template("heart.html")

# Load trained model
model = joblib.load("models/diabetes.pkl")

# SHAP explainer (auto-detect model type)
explainer = shap.TreeExplainer(model)

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    try:
        data = request.get_json()

        # =========================
        # INPUT FEATURES 
        # =========================
        features = pd.DataFrame([{
            "Pregnancies": float(data.get('pregnancies') or 0),
            "Glucose": float(data.get('glucose') or 0),
            "BloodPressure": float(data.get('bp') or 0),
            "SkinThickness": float(data.get('skin') or 0),
            "Insulin": float(data.get('insulin') or 0),
            "BMI": float(data.get('bmi') or 0),
            "DiabetesPedigreeFunction": float(data.get('dpf') or 0),
            "Age": float(data.get('age') or 0)
        }])
        features = features[[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]]

        # =========================
        # MODEL PREDICTION
        # =========================
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1]
        else:
            probability = model.predict(features)[0]

        percentage = round(probability * 100, 2)

        # =========================
        # RISK LABEL
        # =========================
        if percentage < 30:
            result = "Low Risk"
        elif percentage < 50:
            result = "Medium Risk"
        elif percentage < 75:
            result = "High Risk"
        else:
            result = "Very High Risk"

        # =========================
        # SAVE TO DB 
        # =========================
        try:
            user_id = session.get('user_id')
            save_prediction(user_id, "Diabetes", percentage)
        except Exception as db_error:
            print("DB ERROR:", db_error)

        # =========================
        #  SHAP 
        # =========================
        shap_values = explainer.shap_values(features)

        # handle binary classification
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        values = np.array(values).flatten()

        feature_names = [
            "Pregnancies", "Glucose", "BP", "Skin",
            "Insulin", "BMI", "DPF", "Age"
        ]

        abs_vals = np.abs(values)
        total = np.sum(abs_vals)

        # safe normalization
        if total == 0:
            shap_data = {name: 0 for name in feature_names}
        else:
            shap_data = {
                feature_names[i]: round((abs_vals[i] / total) * 100, 2)
                for i in range(len(feature_names))
            }

        # =========================
        # SORT + TOP FEATURES
        # =========================
        sorted_features = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)

        top_features = [f[0] for f in sorted_features[:4]]

        # keep only top 4 for UI
        shap_data = dict(sorted_features[:4])

        # =========================
        # RECOMMENDATIONS
        # =========================
        recommendations = []

        if "Glucose" in top_features:
            recommendations.append("High glucose detected. Consider HbA1c and fasting glucose test.")

        if "BMI" in top_features:
            recommendations.append("Elevated BMI. Lifestyle changes and weight management recommended.")

        if "Age" in top_features:
            recommendations.append("Age-related risk. Regular monitoring is advised.")

        if "Insulin" in top_features:
            recommendations.append("Insulin irregularity detected. Consult endocrinologist.")

        if "BP" in top_features:
            recommendations.append("Blood pressure is a contributing factor. Monitor regularly.")

        if not recommendations:
            recommendations.append("Maintain healthy lifestyle and regular checkups.")

        # =========================
        # FINAL RESPONSE
        # =========================
        return jsonify({
            "result": result,
            "percentage": percentage,
            "shap": shap_data,
            "top_features": top_features,
            "recommendation": " ".join(recommendations)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    

#heart
import joblib

heart_model = joblib.load("models/heart_model.pkl")
heart_columns = joblib.load("models/heart_columns.pkl")
heart_explainer = shap.TreeExplainer(heart_model)

@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    try:
        data = request.get_json()

        # =========================
        # CREATE FEATURES 
        # =========================
        features = pd.DataFrame([{
            "age": float(data.get('age', 0)),
            "sex": float(data.get('sex', 0)),
            "chest_pain_type": float(data.get('cp', 0)),
            "resting_bp": float(data.get('trestbps', 0)),
            "cholesterol": float(data.get('chol', 0)),
            "fasting_blood_sugar": float(data.get('fbs', 0)),
            "rest_ecg": float(data.get('restecg', 0)),
            "max_heart_rate": float(data.get('thalach', 0)),
            "exercise_angina": float(data.get('exang', 0)),
            "st_depression": float(data.get('oldpeak', 0)),
            "slope": float(data.get('slope', 0)),
            "num_major_vessels": float(data.get('ca', 0)),
            "thal": float(data.get('thal', 0))
        }])

        
        features = features.reindex(columns=heart_columns, fill_value=0)

        feature_names = heart_columns

        # =========================
        # MODEL PREDICTION
        # =========================
        proba = heart_model.predict_proba(features)
        # FORCE SCALAR
        probability = float(proba[0][1])
        percentage = float(probability * 100)
        percentage = round(percentage, 2)

        # =========================
        # RISK LABEL
        # =========================
        if percentage < 30:
            result = "Low Risk"
        elif percentage < 60:
            result = "Medium Risk"
        else:
            result = "High Risk"

        # =========================
        # SHAP EXPLAINABILITY
        # =========================
        shap_values = heart_explainer.shap_values(features)

        # HANDLE BOTH CASES
        if isinstance(shap_values, list):
            # binary classification → take class 1
            values = shap_values[1][0]
        else:
            # single output case
            values = shap_values[0]

        # Convert safely
        values = np.array(values).flatten()
        abs_vals = np.abs(values)
        total = float(np.sum(abs_vals))

        # Friendly names
        label_map = {
            "chest_pain_type": "Chest Pain",
            "cholesterol": "Cholesterol",
            "max_heart_rate": "Heart Rate",
            "exercise_angina": "Exercise Angina",
            "st_depression": "ST Depression",
            "resting_bp": "Blood Pressure",
            "fasting_blood_sugar": "Fasting Sugar",
            "num_major_vessels": "Major Vessels",
            "thal": "Thalassemia",
            "rest_ecg": "ECG",
            "age": "Age",
            "sex": "Sex",
            "slope": "Slope"
        }
        
        shap_data = {}

        for i in range(len(heart_columns)):
            val = float(abs_vals[i])

            percent = (val / total) * 100 if total != 0 else 0
            percent = round(float(percent), 2)

            label = label_map.get(heart_columns[i], heart_columns[i])
            shap_data[label] = percent

        # =========================
        # TOP FEATURES
        # =========================
        sorted_features = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:3]]

        # =========================
        # RECOMMENDATIONS
        # =========================
        recommendations = []

        if "Cholesterol" in top_features:
            recommendations.append("Reduce fatty foods and monitor cholesterol.")

        if "Blood Pressure" in top_features:
            recommendations.append("Maintain BP with low-salt diet and exercise.")

        if "Heart Rate" in top_features:
            recommendations.append("Abnormal heart rate detected. Consider checkup.")

        if "Age" in top_features:
            recommendations.append("Age-related risk. Regular screening advised.")

        if not recommendations:
            recommendations.append("Maintain healthy lifestyle and routine checkups.")

        # =========================        
        # SAVE TO DATABASE
        # =========================
        try:
            user_id = session.get('user_id')  # None if guest
            save_prediction(user_id, "Heart Disease", percentage)
        except Exception as db_error:
            print("DB ERROR:", db_error)

        # =========================
        return jsonify({
            "percentage": percentage,
            "result": result,
            "shap": shap_data,
            "top_features": top_features,
            "recommendation": " ".join(recommendations)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})


#liver

import joblib
import numpy as np
from flask import request, jsonify, render_template

# LOAD
liver_model = joblib.load("models/liver_model.pkl")

liver_explainer = joblib.load("models/explainer.pkl")
liver_features = joblib.load("models/feature_names.pkl")


@app.route('/liver')
def liver():
    return render_template("liver.html", features=liver_features)


@app.route('/liver_predict', methods=['POST'])
def liver_predict():
    try:
        data = request.get_json()

        # ----------------------
        if (
            float(data["Total_Bilirubin"]) < 1.2 and
            float(data["Direct_Bilirubin"]) < 0.3 and
            float(data["Alamine_Aminotransferase"]) < 50 and
            float(data["Aspartate_Aminotransferase"]) < 40 and
            float(data["Albumin"]) > 3.5
        ):
            
            #save
            try:
                user_id = session.get('user_id')
                save_prediction(user_id, "Liver Disease", 10.0)
            except Exception as db_error:
                print("DB ERROR:", db_error)
            return jsonify({
                "percentage": 10.0,
                "result": "Low Risk",
                "shap": {},
                "recommendation": "All values are within normal range. Maintain a healthy lifestyle."
            })

        # ----------------------
        #  MODEL INPUT
        # ----------------------
        values = [float(data.get(f, 0)) for f in liver_features]

        import pandas as pd
        features_df = pd.DataFrame([values], columns=liver_features)

        # NO SCALING 
        features_scaled = features_df

        # ----------------------
        # ----------------------
        #  PREDICTION 
        # ----------------------
        probs = liver_model.predict_proba(features_scaled)[0]
        
        #  PROBABILITY 
        disease_prob = float(probs[0])  
        
        percentage = round(disease_prob * 100, 2)
        
        # Threshold logic
        if disease_prob > 0.7:
            result = "High Risk"
        elif disease_prob > 0.4:
            result = "Medium Risk"
        else:
            result = "Low Risk"
        # ----------------------
        #  THRESHOLD LOGIC
        # ----------------------
        if disease_prob > 0.7:
            result = "High Risk"
        elif disease_prob > 0.4:
            result = "Medium Risk"
        else:
            result = "Low Risk"

        # ----------------------
        #  SHAP EXPLANATION
        # ----------------------
        shap_vals = liver_explainer.shap_values(features_df)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]

        vals = np.abs(vals).astype(float)
        total = np.sum(vals)

        shap_data = {
            f: float(round((vals[i] / total) * 100 if total else 0, 2))
            for i, f in enumerate(liver_features)
        }

        
        #save 
        try:
            user_id = session.get('user_id')
            save_prediction(user_id, "Liver Disease", percentage)
        except Exception as db_error:
            print("DB ERROR:", db_error)
        
        # ----------------------
        #  RESPONSE
        # ----------------------
        return jsonify({
            "percentage": float(percentage),
            "result": str(result),
            "shap": shap_data,
            "recommendation": "Maintain a healthy diet and regular liver checkups."
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
    
#parkinsons 

@app.route('/parkinsons')
def parkinsons():
    return render_template('parkinsons.html')

# =========================
# LOAD MODELS
# =========================
parkinsons_model = joblib.load("models/parkinson_model.pkl")     
parkinsons_base = joblib.load("models/parkinson_base.pkl")       #  base model
parkinsons_scaler = joblib.load("models/parkinson_scaler.pkl")

# =========================
# FEATURE NAMES
# =========================
feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

# =========================
# SHAP 
# =========================
parkinsons_explainer = shap.TreeExplainer(parkinsons_base)


@app.route('/parkinsons_predict', methods=['POST'])
def parkinsons_predict():
    try:
        import numpy as np
        import pandas as pd

        # =========================
        # INPUT
        # =========================
        def get_val(name):
            val = request.form.get(name)
            return float(val) if val not in [None, ""] else 0.0

        values = [
            get_val('fo'), get_val('fhi'), get_val('flo'),
            get_val('jitter'), get_val('jitter_abs'),
            get_val('rap'), get_val('ppq'), get_val('ddp'),
            get_val('shimmer'), get_val('shimmer_db'),
            get_val('apq3'), get_val('apq5'),
            get_val('apq'), get_val('dda'),
            get_val('nhr'), get_val('hnr'),
            get_val('rpde'), get_val('dfa'),
            get_val('spread1'), get_val('spread2'),
            get_val('d2'), get_val('ppe')
        ]

        # =========================
        # DATAFRAME + SCALE
        # =========================
        features = pd.DataFrame([values], columns=feature_names)
        scaled = parkinsons_scaler.transform(features)

        # =========================
        # PREDICTION
        # =========================
        prob = parkinsons_model.predict_proba(scaled)[0]
        disease_prob = float(prob[1])

        # smooth probability
        disease_prob = 0.85 * disease_prob + 0.075
        percentage = round(disease_prob * 100, 2)

        # =========================
        # RISK LOGIC
        # =========================
        if disease_prob >= 0.75:
            result = "Parkinson's Detected"
            risk_level = "HIGH"
        elif disease_prob >= 0.45:
            result = "Moderate Risk"
            risk_level = "MEDIUM"
        else:
            result = "Low Risk"
            risk_level = "LOW"

        # =========================
        # SHAP 
        # =========================
        shap_values = parkinsons_explainer.shap_values(scaled)

        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        vals = np.array(vals).flatten()

        abs_vals = np.abs(vals)
        total = np.sum(abs_vals)
        if total == 0:
            total = 1

        shap_dict = {}

        for i, name in enumerate(feature_names):
            percent = (abs_vals[i] / total) * 100
            impact = "positive" if vals[i] > 0 else "negative"

            shap_dict[name] = {
                "value": round(float(percent), 2),
                "impact": impact
            }

        # =========================
        # SORT + TOP FEATURES
        # =========================
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: x[1]["value"],
            reverse=True
        )

        top_features = [f[0] for f in sorted_features[:3]]
        shap_dict = dict(sorted_features[:5])

        # =========================
        # SAVE TO DB
        # =========================
        try:
            user_id = session.get('user_id')
            save_prediction(user_id, "Parkinson's Disease", percentage)
        except:
            pass

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "prediction": result,
            "confidence": percentage,
            "risk_level": risk_level,
            "shap": shap_dict,
            "top_features": top_features
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# TYPHOID MODEL
# =========================
typhoid_model = joblib.load("models/typhoid_model.pkl")

typhoid_features = [
    "Fever_Duration",
    "Abdominal_Pain",
    "Headache",
    "Diarrhea",
    "Weakness",
    "Vomiting",
    "Widal_Test",
    "Typhidot_Test",
    "WBC_Count",
    "Platelet_Count"
]

@app.route('/typhoid')
def typhoid():
    return render_template("typhoid.html")

@app.route('/typhoid_predict', methods=['POST'])
def typhoid_predict():
    try:
        data = request.get_json()

        # =========================
        # CREATE INPUT ARRAY
        # =========================
        values = [
            float(data.get("Fever_Duration", 0)),
            float(data.get("Abdominal_Pain", 0)),
            float(data.get("Headache", 0)),
            float(data.get("Diarrhea", 0)),
            float(data.get("Weakness", 0)),
            float(data.get("Vomiting", 0)),
            float(data.get("Widal_Test", 0)),
            float(data.get("Typhidot_Test", 0)),
            float(data.get("WBC_Count", 0)),
            float(data.get("Platelet_Count", 0))
        ]

        features = pd.DataFrame([values], columns=typhoid_features)

        # =========================
        # PREDICTION
        # =========================
        proba = typhoid_model.predict_proba(features)[0][1]
        percentage = round(float(proba * 100), 2)

        # =========================
        # RISK LABEL
        # =========================
        if percentage < 30:
            result = "Low Risk"
        elif percentage < 60:
            result = "Medium Risk"
        else:
            result = "High Risk"

        # =========================
        #  SHAP 
        # =========================
        explainer = shap.TreeExplainer(typhoid_model)

        shap_values = explainer.shap_values(features)

        values = np.array(shap_values)

        if isinstance(shap_values, list):
            values = shap_values[1]
        elif values.ndim == 3:
            values = values[:, :, 1]

        values = values.flatten()
        abs_vals = np.abs(values)
        total = np.sum(abs_vals)

        shap_data = {}

        for i in range(len(typhoid_features)):
            percent = (abs_vals[i] / total) * 100 if total != 0 else 0
            shap_data[typhoid_features[i]] = round(percent, 2)

        # =========================
        # TOP FEATURES
        # =========================
        sorted_features = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:3]]

        # =========================
        # RECOMMENDATION
        # =========================
        recommendations = []

        recommendations = []

        if percentage < 30:
            recommendations.append("Low risk. Maintain hygiene and healthy lifestyle.")

        else:
            if values[0] > 3:
                recommendations.append("Persistent fever detected.")

            if values[6] == 1:
                recommendations.append("Positive Widal test indicates infection.")

            if values[8] > 10000:
                recommendations.append("High WBC count indicates infection.")
                
            if values[9] < 150000:
                recommendations.append("Low platelet count detected.")
                
        # =========================
        #  SAVE TO DATABASE
        # =========================
        try:
            user_id = session.get('user_id')
            save_prediction(user_id, "Typhoid", percentage)
        except Exception as db_error:
            print("DB ERROR:", db_error)

        # =========================
        # FINAL RESPONSE
        # =========================
        return jsonify({
            "percentage": percentage,
            "result": result,
            "shap": shap_data,
            "top_features": top_features,
            "recommendation": " ".join(recommendations)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    

#chatbot

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")

    reply = get_chat_response(user_message)

    return jsonify({"reply": reply})

# =========================
# AUTH ROUTES
# =========================

@app.route('/signin')
def signin():
    return render_template('signin.html')


@app.route('/signup', methods=['POST'])
def signup():
    return signup_user()


@app.route('/login', methods=['POST'])
def login():
    return login_user()


@app.route('/history')
def history():
    user_id = session.get('user_id')

    if not user_id:
        return redirect('/signin')   # protect

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        SELECT * FROM predictions 
        WHERE user_id=%s 
        ORDER BY created_at DESC
    """, (user_id,))

    data = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template("history.html", data=data)

def save_prediction(user_id, disease, risk):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO predictions (user_id, disease, risk_score)
        VALUES (%s, %s, %s)
    """, (user_id, disease, risk))

    db.commit()
    cursor.close()
    db.close()
    
@app.route('/admin')
def admin():
    if session.get('is_admin') != True:
        return "Unauthorized", 403

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    data = cursor.fetchall()

    return render_template("admin.html", data=data)

@app.route('/delete/<int:id>', methods=['POST'])
def delete_record(id):
    if session.get('is_admin') != True:
        return "Unauthorized", 403

    db = get_db()
    cursor = db.cursor()

    cursor.execute("DELETE FROM predictions WHERE id=%s", (id,))
    db.commit()

    return '', 204


@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')

    if not user_id:
        return redirect('/signin')

    db = get_db()
    cursor = db.cursor()

    # Get last 5 predictions
    cursor.execute("""
        SELECT * FROM predictions 
        WHERE user_id=%s 
        ORDER BY created_at DESC 
        LIMIT 5
    """, (user_id,))
    data = cursor.fetchall()

    #  Get user details
    cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    user_row = cursor.fetchone()

    #  Convert to dictionary 
    columns = [col[0] for col in cursor.description]
    user = dict(zip(columns, user_row)) if user_row else None

    return render_template("dashboard.html", data=data, user=user)

@app.route('/logout')
def logout():
    session.clear()  

    return redirect('/signin')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/update-profile', methods=['POST'])
def update_profile():
    user_id = session.get('user_id')

    if not user_id:
        return redirect('/signin')

    db = get_db()
    cursor = db.cursor()   

    age = request.form.get('age')
    gender = request.form.get('gender')
    blood_group = request.form.get('blood_group')
    height = request.form.get('height')
    weight = request.form.get('weight')
    contact = request.form.get('contact')

    cursor.execute("""
        UPDATE users 
        SET age=%s, gender=%s, blood_group=%s, height=%s, weight=%s, contact=%s
        WHERE id=%s
    """, (age, gender, blood_group, height, weight, contact, user_id))

    db.commit()

    return redirect('/dashboard')

if __name__ == "__main__":
    app.run(debug=True)