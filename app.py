from flask import Flask, request, jsonify
import joblib
import pandas as pd

import joblib

app = Flask(__name__)


# Export the model using joblib
joblib.dump(model, 'logistic_regression_model.joblib')

# Export the scaler using joblib
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler successfully exported in joblib format:")
print("- 'logistic_regression_model.joblib'")
print("- 'scaler.joblib'")


model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0].tolist()

        return jsonify({
            'prediction': int(prediction),
            'probabilities': {
                'no_dropout': probability[0],
                'dropout': probability[1]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
