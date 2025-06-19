from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("knn_model.pkl", "rb"))
normalizer = pickle.load(open("normalizer.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))

# Clean mapping for dropdowns (one-hot columns)
categorical_mappings = {
    'gender': ['male', 'female', 'transgender'],
    'place': ['rural', 'urban', 'unknown'],
    'alcohol_type': ['country liquor', 'branded liquor', 'both'],
    'hbv': ['negative', 'positive'],
    'hcv': ['negative', 'positive'],
    'diabetes': ['no', 'yes'],
    'obesity': ['no', 'yes'],
    'family_history': ['no', 'yes'],
    'usg': ['no', 'yes']
}

# Numeric input fields (remaining)
numeric_fields = [
    'age', 'duration of alcohol consumption(years)', 'quantity of alcohol consumption (quarters/day)',
    'blood pressure (mmhg)', 'tch', 'tg', 'ldl', 'hdl', 'hemoglobin (g/dl)', 'pcv (%)',
    'mcv (femtoliters/cell)', 'total count', 'polymorphs (%)', 'lymphocytes (%)', 'monocytes (%)',
    'eosinophils (%)', 'basophils (%)', 'platelet count (lakhs/mm)', 'total bilirubin (mg/dl)',
    'direct (mg/dl)', 'indirect (mg/dl)', 'total protein (g/dl)', 'albumin (g/dl)', 'globulin (g/dl)',
    'a/g ratio', 'al.phosphatase (u/l)', 'sgot/ast (u/l)', 'sgpt/alt (u/l)'
]
@app.route("/")
def home():
    return render_template("home.html")  # Save the homepage as templates/home.html

@app.route("/predict", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":
        try:
            # 1. Collect numeric values
            data = {field: float(request.form.get(field, 0)) for field in numeric_fields}

            # 2. Create one-hot encoded categorical values
            for field, categories in categorical_mappings.items():
                selected = request.form.get(field)
                for cat in categories:
                    key = f"{field}_{cat}".replace(" ", "_").lower()
                    full_key = None
                    # Match with model feature
                    for mf in model_features:
                        if key in mf.replace(" ", "_").lower():
                            full_key = mf
                            break
                    if full_key:
                        data[full_key] = 1 if cat == selected else 0

            # 3. Align with model features
            df = pd.DataFrame([data])
            df = df.reindex(columns=model_features, fill_value=0)

            # 4. Normalize and predict
            X = normalizer.transform(df)
            pred = model.predict(X)[0]
            prediction = "✅ Cirrhosis Detected" if pred == 1 else "❌ No Cirrhosis Detected"

        except Exception as e:
            prediction = f"⚠️ Error: {str(e)}"

    return render_template("index.html", prediction=prediction, numeric_fields=numeric_fields)

if __name__ == "__main__":
    app.run(debug=True)
