# Import dependencies
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

# confirm if the directory file location is correct
for dirname, _, filenames in os.walk(r"C:\Users\Administrator\Downloads\train.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filenames))

# Load the dataset in a dataframe object and include only four features as mentioned
df = pd.read_csv(r"C:\Users\Administrator\Downloads\train.csv")
include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df_ = df[include]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.items():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")