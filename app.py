from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model and scaler
rf_model = joblib.load('random_forest_model.pkl')
knn_model = joblib.load('knn_model.pkl')
logreg_model = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
url = 'creditcard.csv'
data = pd.read_csv(url)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Only .xlsx files are supported'}), 400

    model_choice = request.form.get('model', 'rf')
    model_map = {
        'rf': rf_model,
        'knn': knn_model,
        'logreg': logreg_model
    }
    model = model_map.get(model_choice, rf_model)

    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()

        # Standardize required columns from training data (keep original case)
        required_columns = [col.strip() for col in data.columns if col.strip().lower() != 'class']

        # Debug: Print columns for troubleshooting
        print("Uploaded file columns:", list(df.columns))
        print("Required columns:", required_columns)

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f"Missing columns in uploaded file: {', '.join(missing_cols)}"}), 400

        # Scale 'Amount'
        df['Amount'] = scaler.transform(df[['Amount']])

        # Reorder columns to match training data
        df = df[required_columns]

        # Debug: Print shape and dtypes
        print("Prediction DataFrame shape:", df.shape)
        print("Prediction DataFrame dtypes:", df.dtypes)

        predictions = model.predict(df)
        df['Prediction'] = predictions
        df['Prediction'] = df['Prediction'].apply(lambda x: 'Fraudulent' if x == 1 else 'Not Fraudulent')

        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        # Save the file to the 'results' folder for later download
        results_folder = os.path.join(app.root_path, 'results')
        os.makedirs(results_folder, exist_ok=True)
        output_path = os.path.join(results_folder, 'prediction_output.xlsx')
        df.to_excel(output_path, index=False)

        return send_file(
            output,
            as_attachment=True,
            download_name='prediction_output.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html', dataset=data.head(10).to_html(classes='table'))

@app.route('/statistics')
def statistics():
    # You can render a statistics page or return JSON as needed
    # For now, let's render a template (create statistics.html)
    # Or you can keep returning JSON if you prefer
    precision = 0.85
    recall = 0.90
    auprc = 0.95

    stats = {
        "Precision": precision,
        "Recall": recall,
        "AUPRC": auprc
    }

    return render_template('statistics.html', stats=stats)

@app.route('/download_latest_results')
def download_latest_results():
    results_folder = os.path.join(app.root_path, 'results')
    filename = 'prediction_output.xlsx'
    return send_from_directory(results_folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

