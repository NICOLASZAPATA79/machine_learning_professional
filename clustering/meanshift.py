
# Importación de modulos de uso general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importación de modulos para clustering
from sklearn.cluster import MeanShift

if __name__ == '__main__':
    df_candy = pd.read_csv('./Data/candy.csv')
    print(df_candy.head(10))

    X = df_candy.drop('competitorname',axis=1)

    meanshift = MeanShift().fit(X)
    print('='*64)
    print(meanshift.labels_)
    print('='*64)
    print(f'La cantidad de grupos creados es: {len(meanshift.cluster_centers_)}')
    df_candy['meanshift_group'] = meanshift.predict(X)
    print(df_candy)


