# Importación de modulos de uso general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importación modulos de ensamble
from sklearn.ensemble import GradientBoostingClassifier

# Importación modulos de split datos
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    df_heart = pd.read_csv('./Data/heart.csv')
    print(df_heart.head(10))
    print(df_heart['target'].value_counts(normalize=True))

    # Split de los datos
    X = df_heart.drop('target',axis=1)
    y = df_heart['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=42)

    # Entrenamiento para la clasifiación con boosting
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train,y_train)
    boost_pred = boost.predict(X_test)
    print('='*64)
    print(f'El accuracy para el modelo con boosting es de: {accuracy_score(y_test,boost_pred)}')

    # Alternativa para identificar el numero de estimadores necesarios para obtener el mejor accuracy

    estimators = range(10, 200, 10)
    total_accuracy = []
    for i in estimators:
        boost_iter = GradientBoostingClassifier(
            n_estimators=i).fit(X_train, y_train)
        boost_iter_pred = boost_iter.predict(X_test)

        total_accuracy.append(accuracy_score(y_test, boost_iter_pred))

    print(np.array(total_accuracy).max())
    plt.plot(estimators, total_accuracy)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.show()



