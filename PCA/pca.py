# Importación de librerias
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Importación modulos especificos

# Importación modulo de clasificación
from sklearn.linear_model import LogisticRegression
# Importación modulo de división de los datos
from sklearn.model_selection import train_test_split
# Importación modulos escalamiento de datos
from sklearn.preprocessing import StandardScaler
# Importación modulo reduccion de dimensionalidad
from sklearn.decomposition  import PCA
from sklearn.decomposition  import IncrementalPCA

# Importación modulo metricas
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_heart = pd.read_csv('./Data/heart.csv') #Llamando con sistema de rutas de windows
    # Impresión Dataframe
    print(df_heart.head(5))

    # Split de los datos en features
    X = df_heart.drop(['target'],axis=1)

    # Split de los datos en target
    y = df_heart['target']

    # Escalamiento de los features
    scaler  = StandardScaler()
    X = scaler.fit_transform(X)

    # Split de los datos, datos de entrenamiento y datos de prueba
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)

    # Nuevas dimensiones resultado del train_test_split
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Verificamos proporción del target

    print(df_heart['target'].value_counts(normalize=True))

    # Configuración algoritmo PCA, el número de componentes por defecto es = min(n_registers/n_features)
    pca = PCA(n_components=5)
    pca.fit(X_train)
    pca_x_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)


     # Configuración algoritmo PCA, el número de componentes por defecto es = min(n_registers/n_features)
    ipca = IncrementalPCA(n_components=5,batch_size=10)
    ipca.fit(X_train)
    ipca_x_train = ipca.transform(X_train)
    ipca_x_test = ipca.transform(X_test)

    # Graficación para verificar la varianza que tiene cada componente principal acumulada (PCA)
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(np.round(var,decimals=4)*100)
    components = list(range(0,5))
    fig = plt.figure(figsize=(8,8))
    for s,d in zip(components,cum_var):
        plt.annotate(np.round(d,decimals=2),xy=(s,d-2.5))
    plt.plot(components,cum_var,color='red',marker='X')
    plt.title('PCA Decomposition')
    plt.xlabel('Number of components')
    plt.ylabel('%Variance')


    # Graficación para verificar la varianza que tiene cada componente principal acumulada (PCA)
    var_ipca = ipca.explained_variance_ratio_
    cum_var_ipca = np.cumsum(np.round(var,decimals=4)*100)
    components_ipca = list(range(0,5))
    fig = plt.figure(figsize=(8,8))
    for s,d in zip(components_ipca,cum_var_ipca):
         plt.annotate(np.round(d,decimals=2),xy=(s,d-2.5))
    plt.plot(components_ipca,cum_var_ipca,color='blue',marker='X')
    plt.title('IPCA Decomposition')
    plt.xlabel('Number of components')
    plt.ylabel('%Variance')

    #Creación Dataframe con datos transformados por PCA

    pca_data_standard = pd.DataFrame(pca_x_train)
    print(pca_data_standard)
    print(pca_data_standard.shape)

    # Clasificación con regresión logistica -PCA

    model = LogisticRegression(solver='lbfgs')
    model.fit(pca_x_train,y_train)
    y_predict_pca = model.predict(pca_x_test)
    print(accuracy_score(y_test,y_predict_pca))

    # Clasificación con regresión logistica -PCA

    model.fit(ipca_x_train,y_train)
    y_predict_ipca = model.predict(ipca_x_test)
    print(accuracy_score(y_test,y_predict_ipca))

    plt.show()















    # Conversión PCA, variables que se van a utilizar









