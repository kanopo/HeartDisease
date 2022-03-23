import pandas as pd

from sklearn.model_selection import train_test_split


"""
    LEGGO I FILE CON PANDAS
"""
cleveland = pd.read_csv("./1 - dati in csv/cleveland.data.csv", names=[
    "id",
    "ccf",
    "age",
    "sex",
    "painloc",
    "painexer",
    "relrest",
    "pncaden",
    "cp",
    "trestbps",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
    "name"
])
hungarian = pd.read_csv("./1 - dati in csv/hungarian.data.csv", names=[
    "id",
    "ccf",
    "age",
    "sex",
    "painloc",
    "painexer",
    "relrest",
    "pncaden",
    "cp",
    "trestbps",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
    "name"
])
long_beach = pd.read_csv("./1 - dati in csv/long-beach-va.data.csv", names=[
    "id",
    "ccf",
    "age",
    "sex",
    "painloc",
    "painexer",
    "relrest",
    "pncaden",
    "cp",
    "trestbps",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
    "name"
])
switzerland = pd.read_csv("./1 - dati in csv/switzerland.data.csv", names=[
    "id",
    "ccf",
    "age",
    "sex",
    "painloc",
    "painexer",
    "relrest",
    "pncaden",
    "cp",
    "trestbps",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
    "name"
])

"""
    SEPARO LA LABEL TARGHET PER USARE train_test_split
"""
y_cleveland = cleveland.pop("num")
X_cleveland = cleveland

y_hungarian = hungarian.pop("num")
X_hungarian = hungarian

y_long_beach = long_beach.pop("num")
X_long_beach = long_beach

y_switzerland = switzerland.pop("num")
X_switzerland = switzerland

"""
    APPLICO TRAIN TEST SPLIT
"""
X_train_cleveland, X_test_cleveland, y_train_cleveland, y_test_cleveland = train_test_split(
    X_cleveland, y_cleveland, test_size=0.20
)

X_train_hungarian, X_test_hungarian, y_train_hungarian, y_test_hungarian = train_test_split(
    X_hungarian, y_hungarian, test_size=0.20
)

X_train_long_beach, X_test_long_beach, y_train_long_beach, y_test_long_beach = train_test_split(
    X_long_beach, y_long_beach, test_size=0.20
)

X_train_switzerland, X_test_switzerland, y_train_switzerland, y_test_switzerland = train_test_split(
    X_switzerland, y_switzerland, test_size=0.20
)

"""
    RICOSTRUISCO I FILE DI PARTENZA MA QUESTA VOLTA DIVISI IN VALIDATION E IL RESTO(TRAIN E TEST CHE USERÒ PER I MODELLI)
"""
train_cleveland = pd.concat([X_train_cleveland, y_train_cleveland], axis=1)
test_cleveland = pd.concat([X_test_cleveland, y_test_cleveland], axis=1)

train_hungarian = pd.concat([X_train_hungarian, y_train_hungarian], axis=1)
test_hungarian = pd.concat([X_test_hungarian, y_test_hungarian], axis=1)

train_long_beach = pd.concat([X_train_long_beach, y_train_long_beach], axis=1)
test_long_beach = pd.concat([X_test_long_beach, y_test_long_beach], axis=1)

train_switzerland = pd.concat([X_train_switzerland, y_train_switzerland], axis=1)
test_switzerland = pd.concat([X_test_switzerland, y_test_switzerland], axis=1)

"""
    SALVO I NUOVI FILE
"""
train_cleveland.to_csv("./2 - a - train labeled/train-cleveland.csv", index=False)
test_cleveland.to_csv("./2 - b - test labeled/test-cleveland.csv", index=False)

train_hungarian.to_csv("./2 - a - train labeled/train-hungarian.csv", index=False)
test_hungarian.to_csv("./2 - b - test labeled/test-hungarian.csv", index=False)

train_long_beach.to_csv("./2 - a - train labeled/train-long_beach.csv", index=False)
test_long_beach.to_csv("./2 - b - test labeled/test-long_beach.csv", index=False)

train_switzerland.to_csv("./2 - a - train labeled/train-switzerland.csv", index=False)
test_switzerland.to_csv("./2 - b - test labeled/test-switzerland.csv", index=False)
