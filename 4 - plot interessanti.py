import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt

data = pd.read_csv("./3 - first cleanup/train.csv")



def plot_eta(dataset):
    color_palette = [
        '#226F54',
        '#5BC3EB',
        '#FCCA46',
        '#FBCAEF',
        '#DA2C38'
    ]
    pd.crosstab(dataset.age, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)

    plt.title("Frequenza di malattia cardiaca in relazione all'età")
    plt.xlabel('Età')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.sex, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title('Frequenza malattia cardiaca in base al sesso')
    plt.xlabel('Sesso (0 = Donna, 1 = Uomo)')
    plt.xticks(rotation=0)
    plt.legend(["Sano", "Malato 1", "Malato 2", "Malato 3", "Malato 4"])
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.cp, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione alla tipologia del dolore")
    plt.xlabel('Tipologia (1 = angina tipica, 2 = angina atipica, 3 = non angina, 4 = asintomatico  )')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.trestbps, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione alla pressione a riposo")
    plt.xlabel('Pressione a riposo')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.fbs, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione zuccheri nel sangue a digiuno maggiori di 120")
    plt.xlabel('FBS(0 = < 120, 1 = > 120)')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.restecg, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione ai risultati dell'ecg")
    plt.xlabel('ECG(0 = normale, 1 = anomalia, 2 = probabile ipertensione ventricolare)')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.exang, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione alla comparsa dell'angina indotta da sforzo")
    plt.xlabel('EXANG(angina indotta = 1, angina non indotta = 0)')
    plt.ylabel('Frequenza')
    plt.show()

    pd.crosstab(dataset.location, dataset.num).plot(kind="bar", figsize=(16, 9), color=color_palette)
    plt.title("Frequenza di malattia cardiaca in relazione alla città")
    plt.xlabel('Città(0 = cleveland, 1 = hungarian, 2 = long beach, 3 = switzerland)')
    plt.ylabel('Frequenza')
    plt.show()

plot_eta(data)


