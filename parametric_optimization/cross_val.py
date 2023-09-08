# Importación modulos de uso general
import numpy as np
import pandas as pd

# Importación librerías de arboles de decisión para regresión
from sklearn.tree import DecisionTreeRegressor

# Importación del modelo de división de datos o split
from sklearn.model_selection import cross_val_score, KFold

# Importación modulos de metricas de regresion

from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    df_happy = pd.read_csv('./Data/felicidad.csv')
    print(df_happy.head(10))

    # Split de los datos

    X = df_happy.drop(['country','rank','score'],axis=1)
    y = df_happy['score']

    # Entrenamiento de regresión con division por cross_validation

    model = DecisionTreeRegressor()
    score = cross_val_score(model,X,y,cv=3,scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    # División de los datos por Kfolds

    kf = KFold(n_splits=3,shuffle=True,random_state=42 )
    mse_values = []
    r2_scores = []
    for train, test in kf.split(df_happy):
        print(train)
        print(test)

        # Entrenamiento de modelo con kfold

        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test= y.iloc[test]

        model = DecisionTreeRegressor().fit(X_train,y_train)
        predictions = model.predict(X_test)

        mse_values.append(mean_squared_error(y_test,predictions))
        r2_scores.append(r2_score(y_test,predictions))
        print(f'El error cuadratico medio es de  : {np.mean(mse_values)}')
        print(f'Los r2 score es de : {np.mean(r2_scores)}')

