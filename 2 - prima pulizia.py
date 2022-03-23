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

data = pd.read_csv("./2 - one dataset/TRAIN-dataset-heart-disease.csv")
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

data.to_csv("./3 - first cleanup/train.csv", index=False)




