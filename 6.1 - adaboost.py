import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./4 - data cleaning/train.csv")
y = data.pop("target")
X = data


def gradient_boosting(X, y, balanced):
    if balanced:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)

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

        abc.fit(X_train, y_train)

        y_pred = abc.predict(X_validation)

        #print(classification_report(y_validation, y_pred))


    # confronto con il test set
    test = pd.read_csv("./4 - data cleaning/test.csv")

    y_test = test.pop("target")
    X_test = test

    avg = np.mean(X_test, axis=0)
    std = np.std(X_test, axis=0)
    X_test = (X_test - avg) / std
    y_test_pred = abc.predict(X_test)

    valori = [0, 0, 0]
    if balanced:
        #valori[0] = (accuracy_score(y_test, y_test_pred))
        valori[0] = (precision_score(y_test, y_test_pred, average='weighted'))
        valori[1] = (recall_score(y_test, y_test_pred, average='weighted'))
        valori[2] = (f1_score(y_test, y_test_pred, average='weighted'))
    else:
        valori[0] = (precision_score(y_test, y_test_pred, average='weighted'))
        valori[1] = (recall_score(y_test, y_test_pred, average='weighted'))
        valori[2] = (f1_score(y_test, y_test_pred, average='weighted'))

    return valori


print("--------- NON BALANCED -----------")
valori = gradient_boosting(X, y, False)
print("------------------- test ----------------")
print("precision:")
print(valori[0])
print("recall:")
print(valori[1])
print("f1:")
print(valori[2])

color_palette = [
        '#226F54',
        '#5BC3EB',
        '#FCCA46',

    ]


people = ("precision", "recall", "f1")
y_pos = np.arange(len(people))
valori[0] = valori[0] * 100
valori[1] = valori[1] * 100
valori[2] = valori[2] * 100
performance = valori

fig, ax = plt.subplots()

hbars = ax.barh(y_pos, performance, align='center', color=color_palette)
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('Adaboost')

# Label with specially formatted floats
ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(right=110)  # adjust xlim to fit labels

plt.show()


smote = SMOTE()

X_bal, y_bal = smote.fit_resample(X, y)


print("--------- BALANCED -----------")
gradient_boosting(X_bal, y_bal, True)
print("------------------- test ----------------")


print("precision:")
print(valori[0])
print("recall:")
print(valori[1])
print("f1:")
print(valori[2])


"""
    Anche l'albero sembra molto bravo, da stare attenti per l'overfitting
"""

