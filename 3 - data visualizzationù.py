import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

cleveland = pd.read_csv("./3 - first cleanup/train-cleveland.csv")
hungarian = pd.read_csv("./3 - first cleanup/train-hungarian.csv")
switzerland = pd.read_csv("./3 - first cleanup/train-switzerland.csv")
long_beach = pd.read_csv("./3 - first cleanup/train-long_beach.csv")


def data_visualization_pca(dataset, nome):

    dataset = dataset.dropna()
    y = dataset.pop("num")
    X = dataset

    # NORMALIZZO

    X_norm = (X - X.min()) / (X.max() - X.min())

    X_norm_senza_nan = X_norm.dropna(axis=1)

    pca = PCA(n_components=2)  # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm_senza_nan))
    plt.scatter(transformed[y == 0][0], transformed[y == 0][1], label='Sano', c='green')
    plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Malato 1', c='violet')
    plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Malato 2', c='orange')
    plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Malato 3', c='red')
    plt.scatter(transformed[y == 4][0], transformed[y == 4][1], label='Malato 4', c='purple')

    plt.legend(loc=4)
    plt.title("Rappresentazioni tramite PCA(raw data) " + nome)
    plt.show()


data_visualization_pca(cleveland, "cleveland")
#data_visualization_pca(hungarian, "hungarian")
#data_visualization_pca(long_beach, "long_beach")
data_visualization_pca(switzerland, "switzerland")