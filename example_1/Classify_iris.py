import pandas as pd

# Data Collection
from sklearn.datasets import load_iris
# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
print(data.head())
#output :
    #    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  species
    # 0                5.1               3.5                1.4               0.2        0
    # 1                4.9               3.0                1.4               0.2        0
    # 2                4.7               3.2                1.3               0.2        0
    # 3                4.6               3.1                1.5               0.2        0
    # 4                5.0               3.6                1.4               0.2        0
# Check for missing values
print(data.isnull().sum())
#output :
    # sepal length (cm)    0
    # sepal width (cm)     0
    # petal length (cm)    0
    # petal width (cm)     0
    # species              0
    # dtype: int64

# Data Splitting
from sklearn.model_selection import train_test_split
X = data.drop('species' , axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=22, stratify=y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#output :
    # (120, 4)
    # (120,)
    # (30, 4)
    # (30,)
print(X_train.head())
#output :
    #      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # 55                 5.7               2.8                4.5               1.3
    # 5                  5.4               3.9                1.7               0.4
    # 149                5.9               3.0                5.1               1.8
    # 74                 6.4               2.9                4.3               1.3
    # 37                 4.9               3.6                1.4               0.1

# Model Selection
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# Model Training
model.fit(X_train, y_train)

# Model Evaluation
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(y_pred) #output : [1 2 0 0 0 0 0 2 1 0 1 2 1 1 2 2 0 2 0 1 1 1 1 0 2 2 2 0 1 2]
accuracy = accuracy_score(y_test, y_pred)
print(round(accuracy, 2)) #output:  0.97

# Model Deployment
from sklearn.pipeline import make_pipeline 
import joblib

create_model = make_pipeline(model)
joblib.dump(create_model,'example_1/iris_model.pkl')
