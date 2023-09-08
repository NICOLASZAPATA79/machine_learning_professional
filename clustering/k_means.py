
# Importación de modulos de uso general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importación de modulos para clustering
from sklearn.cluster import MiniBatchKMeans


if __name__ == '__main__':
    # Importación y verificación de las features del dataset
    df_candy = pd.read_csv('./Data/candy.csv')
    print(df_candy.head(5))
    print(df_candy.shape)
    print(df_candy.info())
    print(df_candy.describe())
    print(df_candy.isnull().sum())

    # Split de los datos
    X = df_candy.drop('competitorname',axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4,batch_size=8,n_init='auto').fit(X)
    print(f'La cantidad de centros es : {len(kmeans.cluster_centers_)}')
    print('='*64)
    print(kmeans.predict(X))
    df_candy['group']= kmeans.predict(X)
    print(df_candy)
    sns.pairplot(data =df_candy,hue='group')
    plt.show()






