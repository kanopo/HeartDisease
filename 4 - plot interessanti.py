import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./3 - first cleanup/train.csv")



def plot_eta(dataset):
    pd.crosstab(dataset.age, dataset.num).plot(kind="bar", figsize=(20, 6), color=[
        '#226F54',
        '#5BC3EB',
        '#FCCA46',
        '#FBCAEF',
        '#DA2C38'
    ])

    plt.title("Frequenza di malattia cardiaca in relazione all'età")
    plt.xlabel('Età')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.sex, dataset.num).plot(kind="bar", figsize=(15, 6), color=[
        '#226F54',
        '#5BC3EB',
        '#FCCA46',
        '#FBCAEF',
        '#DA2C38'
    ])
    plt.title('Frequenza malattia cardiaca in base al sesso')
    plt.xlabel('Sesso (0 = Donna, 1 = Uomo)')
    plt.xticks(rotation=0)
    plt.legend(["Sano", "Malato 1", "Malato 2", "Malato 3", "Malato 4"])
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.chol, dataset.num).plot(kind="bar", figsize=(20, 6), color=[
        '#226F54',
        '#5BC3EB',
        '#FCCA46',
        '#FBCAEF',
        '#DA2C38'
    ])

    plt.title("Frequenza di malattia cardiaca in relazione al colesterolo")
    plt.xlabel('Colesterolo')
    plt.ylabel('Frequenza')
    plt.show()

plot_eta(data)


