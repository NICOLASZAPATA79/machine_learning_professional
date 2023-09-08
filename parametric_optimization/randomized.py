# Importación modulos de uso general
import numpy as np
import pandas as pd


# Importación del modelo de división de datos o split
from sklearn.model_selection import RandomizedSearchCV

# Importación modulos de metricas de regresion
from sklearn.ensemble import RandomForestRegressor

# Importación metricas de regresión
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    df_happy = pd.read_csv('./Data/felicidad.csv')
    print(df_happy.head(10))

    # Split de los datos

    X = df_happy.drop(['country','rank','score'],axis=1)
    y = df_happy['score'].squeeze()

    # Creación modelo de regresión
    reg = RandomForestRegressor()

    # Creación grilla de parametros para RandomizedsearchCV

    parameters = {
        'n_estimators':range(4,16),
        'criterion': ['squared_error','absolute_error'],
        'max_depth': range(2,11)
    }

    # Creación del estimador aleatorio con los parametros

    rand_est = RandomizedSearchCV(reg,parameters,cv=3,n_iter=10,scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(np.abs(rand_est.best_score_))
    predictions = rand_est.predict(X)
    print(f'El error cuadratico medio es de: {mean_squared_error(y,predictions)}')
    print(f'El R2 score es de: {r2_score(y,predictions)}')
    print(rand_est.predict(X.iloc[[0]]))


