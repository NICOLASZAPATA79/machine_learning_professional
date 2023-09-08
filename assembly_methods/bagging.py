# Importación de modulos de uso general
import pandas as pd


# Importación modulos de ensamble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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


    # Creación modelo de clasifiación con KNeighbors
    knn_class = KNeighborsClassifier().fit(X_train,y_train)
    knn_predictions = knn_class.predict(X_test)
    print('='*64)
    print(f'El accuracy para el modelo de Kneighbors es de : {accuracy_score(y_test,knn_predictions)}')

    # Creación modelo de clasifiación con KNeighbors
    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(),n_estimators=50).fit(X_train,y_train)
    bag_predictions = bag_class.predict(X_test)
    print('='*64)
    print(f'El accuracy para el modelo de bagging es de : {accuracy_score(y_test,bag_predictions)}')


