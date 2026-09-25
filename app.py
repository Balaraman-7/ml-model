from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'ml-model'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([{
            'attendance_percentage': data['attendance_percentage'],
            'cgpa': data['cgpa'],
            'backlog_count': data['backlog_count'],
            'internal_marks': data['internal_marks']
        }])

        scaled_features = scaler.transform(input_df)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]

        return jsonify({
            'prediction': int(prediction),
            'probabilities': {
                'no_dropout': float(probability[0]),
                'dropout': float(probability[1])
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
