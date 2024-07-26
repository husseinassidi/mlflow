import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Load the dataset
data = pd.read_csv('C:/Users/user/Desktop/mlflow/titanic/train.csv')
# Preprocess the data
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Train and log the model
with mlflow.start_run(run_name="Decision Tree Model 2") as run:
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Log parameters and metrics
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "DecisionTreeClassifierModel2")

    # vizualization
    plt.figure(figsize=(20,10))
    plot_tree(model,feature_names=X.columns, class_names=['Not SUrvived','Survived'],filled=True,rounded=True)
    plt.savefig("model2_tree.png")
    plt.show()