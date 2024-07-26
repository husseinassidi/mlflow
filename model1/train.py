import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")

data = pd.read_csv('C:/Users/user/Desktop/mlflow/titanic/train.csv'  )
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
X = data[['Pclass', 'Sex', 'Age', 'Fare']]


y = data['Survived']
with mlflow.start_run(run_name="Decision Tree Model 1") as run:
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "DecisionTreeClassifierModel1")
    plt.figure(figsize=(4,2))
    plot_tree(model,feature_names=X.columns, class_names=['Not SUrvived','Survived'],filled=True,rounded=True)
    plt.savefig("model1_tree.png")
    plt.show()

    