from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(_name_)

# Load the model
model_name = "DecisionTreeClassifierModel1"
latest_version = mlflow.get_latest_versions(model_name, stages=["None"])[0].version
model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = pd.DataFrame(data)
    predictions = model.predict(X)
    return jsonify(predictions.tolist())

if _name_ == '_main_':
    app.run(port=5001)