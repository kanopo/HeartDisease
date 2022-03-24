import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./4 - data cleaning/train.csv")

y = data.pop("target")
X = data

def hard_voting(X, y, balanced):
    if balanced:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    bc = BaggingClassifier(n_estimators=5, n_jobs=-1)
    dtc = DecisionTreeClassifier(max_depth=10)
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    logreg = LogisticRegression(solver='saga', max_iter=6000, multi_class="multinomial", class_weight="balanced", n_jobs=-1)
    rfc = RandomForestClassifier(max_depth=8)
    sgdc = SGDClassifier(n_jobs=-1, class_weight='balanced', early_stopping=True, max_iter=5000)

    vc = VotingClassifier(estimators=[
        ("AdaBoostClassifier", abc),
        ("BaggingClassifier", bc),
        ("DecisionTreeClassifier", dtc),
        ("GradientBoostingClassifier", gbc),
        ("LogisticRegression", logreg),
        ("RandomForestClassifier", rfc)
        #("StocasticGradientDescenderClassifier", sgdc),
    ], voting="soft")

    return cross_val_score(vc, X_train, y_train, cv=10, n_jobs=-1)


print("--------- NON BALANCED -----------")
print(hard_voting(X, y, False))
smote = SMOTE()

print("--------- BALANCED -----------")
X_bal, y_bal = smote.fit_resample(X, y)
print(hard_voting(X_bal, y_bal, True))
