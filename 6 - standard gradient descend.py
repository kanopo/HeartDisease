import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold

data = pd.read_csv("./4 - data cleaning/train.csv")
y = data.pop("target")
X = data


def gradient_boosting(X, y, balanced):
    if balanced:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    sgdc = SGDClassifier(n_jobs=-1, class_weight='balanced', early_stopping=True, max_iter=5000)

    kf = KFold(n_splits=10, random_state=None, shuffle=True)  # initialize KFold
    for train_index, validation_index in kf.split(X):
        # print("TRAIN:", train_index, "VALIDATION:", validation_index)
        X_train = X[train_index[0]:train_index[-1]]
        X_validation = X[validation_index[0]:validation_index[-1]]
        y_train = y[train_index[0]:train_index[-1]]
        y_validation = y[validation_index[0]:validation_index[-1]]
        # Now train the model and take note of its performance

        avg = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - avg) / std
        X_validation = (X_validation - avg) / std

        sgdc.fit(X_train, y_train)

        y_pred = sgdc.predict(X_validation)

        print(classification_report(y_validation, y_pred))

        return y_pred



print("--------- NON BALANCED -----------")
gradient_boosting(X, y, False)

smote = SMOTE()

X_bal, y_bal = smote.fit_resample(X, y)


print("--------- BALANCED -----------")
gradient_boosting(X_bal, y_bal, True)

"""
    non mi fa impazzire
"""