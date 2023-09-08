
# Importación de librerias

import pandas as pd
import sklearn

# Importación librerías de regresión lineal, Lasso, Ridge

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# Importación librería split datos
from sklearn.model_selection import train_test_split

# Importación de librerías de medición
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Importación del dataset

if __name__ == '__main__':
    df_happy = pd.read_csv('./Data/felicidad.csv')
    print(df_happy.head(10))
    print(df_happy.describe())

    # Split de los datos

    X = df_happy[['gdp','family','lifexp','freedom','generosity','corruption','dystopia']]
    y = df_happy['score']

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    # Creación modelos

    # Modelo lineal
    model_linear = LinearRegression().fit(X_train,y_train)
    y_predict_linear = model_linear.predict(X_test)

    # Modelo Lasso
    model_lasso = Lasso(alpha=0.02).fit(X_train,y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    # Modelo Ridge
    model_ridge = Ridge(alpha=1).fit(X_train,y_train)
    y_predict_ridge = model_ridge.predict(X_test)

    # Modelo Elastic net
    model_elasticnet = ElasticNet(alpha=0.05).fit(X_train,y_train)
    y_predict_elasticnet = model_elasticnet.predict(X_test)

    # Evaluación errores-perdida

    linear_loss = mean_squared_error(y_test,y_predict_linear)
    print(f'Error cuadratico medio en regresion lineal: {linear_loss}')
    print(f'El R2 en regresión lineal es de: {r2_score(y_test,y_predict_linear)}')

    lasso_loss = mean_squared_error(y_test,y_predict_lasso)
    print(f'Error cuadratico medio en Lasso: {lasso_loss}')
    print(f'El R2 en Lasso es de: {r2_score(y_test,y_predict_lasso)}')

    ridge_loss = mean_squared_error(y_test,y_predict_ridge)
    print(f'Error cuadratico medio en Ridge: {ridge_loss}')
    print(f'El R2 en Ridge es de: {r2_score(y_test,y_predict_ridge)}')

    elasticnet_loss = mean_squared_error(y_test,y_predict_elasticnet )
    print(f'Error cuadratico medio en Elastic net: {elasticnet_loss}')
    print(f'El R2 en Elastic net es de: {r2_score(y_test,y_predict_elasticnet)}')

    # Impresión de los coeficientes que asigna cada uno de los modelos

    print(f'Los coeficientes en Lasso son: {model_lasso.coef_}')

    print('='*32)

    print(f'Los coeficientes en Ridge son: {model_ridge.coef_}')

    print('='*32)

    print(f'Los coeficientes en Elastic net son: {model_elasticnet.coef_}')






