#создай здесь свой индивидуальный проект!
import pandas as pd

df = pd.read_csv(r"d:\PROGRAMS\PYton\proj\Algoritmika\sky_net\lesson2\train.csv")
df.drop(['bdate','has_photo', 'has_mobile', 'city','education_form','langs','occupation_name',
                'id', 'last_seen', 'career_end', 'career_start'], axis = 1, inplace = True)

#print(df['education_status'].value_counts())
df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
df[list(pd.get_dummies(df['life_main']).columns)] = pd.get_dummies(df['life_main'])
df[list(pd.get_dummies(df['people_main']).columns)] = pd.get_dummies(df['people_main'])
df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])

df.drop(['education_status','life_main', 'people_main', 'occupation_type',], axis = 1, inplace = True)

df.info()
#print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

y = df['result']
X = df.drop('result', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
 
