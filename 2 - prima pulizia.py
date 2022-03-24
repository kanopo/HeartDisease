import os


import pandas as pd

"""
    Leggendo la descrizione el data set posso eliminare alcune features subito:
    - id
    - ccf
    - dummy
    - restckm
    - exerckm
    - thalsev
    - thalpul
    - earlobe
    - lvx1
    - lvx2
    - lvx3
    - lvx4
    - lvf
    - cathef
    - junk
    - name
"""

train = pd.read_csv("./2 - one dataset/TRAIN-dataset-heart-disease.csv")
test = pd.read_csv("./2 - one dataset/TEST-dataset-heart-disease.csv")

def first_clean(data, name):
    data.drop(inplace=True, axis='columns', labels=[
        "id",
        "ccf",
        "dummy",
        "restckm",
        "exerckm",
        "thalsev",
        "thalpul",
        "earlobe",
        "lvx1",
        "lvx2",
        "lvx3",
        "lvx4",
        "lvf",
        "cathef",
        "junk",
        "name",
    ])

    data.to_csv("./3 - first cleanup/" + name + ".csv", index=False)

first_clean(train, "train")
first_clean(test, "test")





