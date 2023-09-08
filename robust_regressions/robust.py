# Importación de librerias
import pandas as pd
import matplotlib.pyplot as plt

# Importación modulos regresion robusta
from sklearn.linear_model import RANSACRegressor, HuberRegressor

# Importación libreria de split train_test
from sklearn.model_selection import train_test_split

# Importación modulo suport vector machines-support vector regressor
from sklearn.svm import SVR

# Importar modulos de metricas
from sklearn.metrics import mean_squared_error, r2_score

# Importar modulo de errores
import warnings
warnings.simplefilter('ignore')

if __name__ == '__main__':
    df_happy = pd.read_csv('./Data/felicidad_corrupt.csv')
    print(df_happy.head(10))
    print(df_happy.shape)

    X = df_happy.drop(['country','rank','score'],axis=1)
    y = df_happy['score']

    # Split de los datos de entrenamiento y test

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3,random_state=42)

    # Creación de diccionario con estimadores

    estimators = {
        'SVR':SVR(gamma='auto',C=1.0,epsilon=0.1),
        'RANSAC':RANSACRegressor(), # Este es un metaestimador, por defecto utiliza una regresión lineal
        'HUBER':HuberRegressor(epsilon=1.35) # Por defecto epsilon es 1.35
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)
        print('='*64)
        print(f'MSE de {name} = {mean_squared_error(y_test,prediction)}')
        print('='*64)
        print(f'R2 score de {name} = {r2_score(y_test,prediction)}')
        plt.xlabel('Real score')
        plt.ylabel('Prediction score')
        plt.title(f'Real scores vs Prediction scores of {name}')
        plt.scatter(y_test,prediction)
        plt.plot(prediction,prediction,color='red',linestyle= '--',label=(f'{name}'))
        plt.legend()
        plt.show()



