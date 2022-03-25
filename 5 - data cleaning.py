import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression

data = pd.read_csv("./3 - first cleanup/train.csv")


def cleaning(data):
    print("------------------- non uniche -------------------")
    for i in range(data.shape[1]):
        print(i, len(np.unique(data.iloc[:, i])))

    """
        LA VARIABILE 5(PNCADEN) È LUNICA COSTANTE
    """

    data.drop(inplace=True, axis='columns', labels=[
        "pncaden"
    ])

    # NEL DATA SET VENGONO USATI I -9 COME VALORI NULLI QUINDI LI SOSTITUISCO CON DEI VERI VALORI NULLI CHE NON
    # FANNO CASINO CON PANDAS
    data.replace(-9, np.nan, inplace=True)

    # DROPPO SE RIGHE O COLONNE HANNO PIÙ DEL 15% DI VALORI NULLI

    print("--------------------- nulli -----------------------")

    for i in data.columns:
        print(i, data[i].isnull().sum())

    perc_drop_na = 15.0

    min_count_cols = int(((100 - perc_drop_na) / 100) * data.shape[0] + 1)
    data = data.dropna(axis=1,
                       thresh=min_count_cols)  # droppo colonne che hanno meno valori non nulli rispetto alla tresh

    min_count_raws = int(((100 - perc_drop_na) / 100) * data.shape[1] + 1)
    data = data.dropna(axis=0,
                       thresh=min_count_raws)  # droppo righe che hanno meno valori non nulli rispetto alla tresh

    numeric_columns = [
        "age",
        "trestbps",
        "chol",
        "thaldur",
        "met",
        "thalach",
        "thalrest",
        "tpeakbps",
        "tpeakbpd",
        "trestbpd",
        "oldpeak",
        "proto",

    ]

    # RIMUOVO TUTTI GLI ESEMPI CHE HANNO UNA DEVIAZIONE MAGGIORE DI 3(OLTRE IL 99%
    for column in numeric_columns:
        data['z'] = np.abs(stats.zscore(data[column], nan_policy='omit'))
        data.drop(data[data['z'] > 3].index, inplace=True)

    data.pop("z")

    data = data.rename(columns={'num': 'target'})

    """
        PER I MODELLI NON MI INTERESSANO LE DATE NELLE QUALI SONO STATE EFFETTUATE LE ANALISI
        MA MI INTERESSANO SOLO LE ANALISI
    """

    data.drop(inplace=True, axis='columns', labels=[
        "ekgmo",  # date degli esami, non interessanti
        "ekgday",
        "ekgyr",
        "cmo",  # sempre date
        "cday",
        "cyr"
    ])

    """
        proto è una variabile categorica che dovrebbe rappresentare il protocollo utilizzato per l'esercizio
        sotto sforzo, solo che spesso i valori sono incoerenti con le 12 categorie del problema.
        ho decisono di non considerare la features per non dorppare magari il 40 % degli esempi
    """
    print(data["proto"].describe())
    data.drop(inplace=True, axis='columns', labels=[
        "proto"
    ])

    data = data.astype({
        "age": int,
        "sex": bool,
        "cp": 'category',
        "trestbps": float,
        "htn": bool,
        "chol": float,
        "fbs": bool,
        "restecg": 'category',
        "dig": bool,
        "prop": bool,
        "nitr": bool,
        "pro": bool,
        "diuretic": bool,
        "thaldur": int,
        "met": float,
        "thalach": int,
        "thalrest": int,
        "tpeakbps": int,
        "tpeakbpd": int,
        "trestbpd": int,
        "exang": bool,
        "xhypo": bool,
        "oldpeak": float,
        "location": 'category',
        "target": 'category'

    }, errors="ignore")

    data = data.fillna(data.mean())
    data.dropna(inplace=True)

    y = data.pop("target")
    X = data

    """
        Elimino le features che hanno uno score con f_classif basso:
    """
    data.drop(inplace=True, axis='columns', labels=[
        "tpeakbpd",
        "dig",
    ])

    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    print("F_CLASSIF")
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.sort_values(ascending=False, by=['Score']))

    """
        Elimino le features che hanno uno score con mutual_info_regression basso:
    """
    data.drop(inplace=True, axis='columns', labels=[
        "xhypo",
        "htn",
        "diuretic",
        "trestbpd",
        "thalrest",
        "nitr",
        "tpeakbps",
        "prop",
        "pro",
    ])
    bestfeatures = SelectKBest(score_func=mutual_info_regression, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    print("MUTUAL INFO REGRESSION")
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.sort_values(ascending=False, by=['Score']))

    print(data)

    print(data.describe())

    final = pd.concat([X, y], axis=1)
    return final


final = cleaning(data)
final.to_csv("./4 - data cleaning/train.csv", index=False)

test = pd.read_csv("./3 - first cleanup/test.csv")
final_test = cleaning(test)
final.to_csv("./4 - data cleaning/test.csv", index=False)

