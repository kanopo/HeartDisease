import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


data = pd.read_csv("./3 - first cleanup/train.csv")


y = data.pop("num")
X = data
X.pop("location")
# NORMALIZZO

X_norm = (X - X.min()) / (X.max() - X.min())

X_norm_senza_nan = X_norm.dropna(axis=1)

pca = PCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm_senza_nan))
plt.scatter(transformed[y == 0][0], transformed[y == 0][1], label='Sano', c='#226F54')
plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Malato 1', c='#5BC3EB')
plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Malato 2', c='#FCCA46')
plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Malato 3', c='#FBCAEF')
plt.scatter(transformed[y == 4][0], transformed[y == 4][1], label='Malato 4', c='#DA2C38')

plt.legend(loc=4)
plt.title("Rappresentazioni tramite PCA(raw data)")
plt.show()

lda = LDA(n_components=2, ) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm_senza_nan, y))

# Plot all three series
plt.scatter(lda_transformed[y == 0][0], lda_transformed[y == 0][1], label='Sano', c='#226F54')
plt.scatter(lda_transformed[y == 1][0], lda_transformed[y == 1][1], label='Malato 1', c='#5BC3EB')
plt.scatter(lda_transformed[y == 2][0], lda_transformed[y == 2][1], label='Malato 2',  c='#FCCA46')
plt.scatter(lda_transformed[y == 3][0], lda_transformed[y == 3][1], label='Malato 3', c='#FBCAEF')
plt.scatter(lda_transformed[y == 4][0], lda_transformed[y == 4][1], label='Malato 4', c='#DA2C38')

# Display legend and show plot
plt.legend(loc=4)
plt.title("Rappresentazioni tramite LDA(raw data)")
plt.show()
