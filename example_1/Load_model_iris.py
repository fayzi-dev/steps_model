import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
X = data.drop('species' , axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=22, stratify=y)


#Load Model
model = joblib.load('example_1/iris_model.pkl')
# print(model)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)  
# Output:
    # [1 2 0 0 0 0 0 2 1 0 1 2 1 1 2 2 0 2 0 1 1 1 1 0 2 2 2 0 2 2] 