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

my_fyles = os.listdir("./2 - a - train labeled")

for file in my_fyles:
    data = pd.read_csv("./2 - a - train labeled/" + str(file))
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

    data.to_csv("./3 - first cleanup/" + str(file), index=False)




