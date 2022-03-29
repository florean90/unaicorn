import pandas as pd


df = pd.read_csv('train.csv')
df.loc[df['Fare'] > 400, 'Fare'] = df['Fare'].median()
df.loc[df['Age'] > 70, 'Age'] = 70
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
del df['Cabin']


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'No title in name'
titles = set([x for x in df.Name.map(lambda x: get_title(x))])


def shorter_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'Dona', 'the Countess', 'Lady', 'Sir']:
        return 'Royalty'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else: return title


df['Title'] = df['Name'].map(lambda x: get_title(x))
df['Title'] = df.apply(shorter_titles, axis=1)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.Sex.replace(('male', 'female'), (0,1), inplace=True)
df.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)
df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0,1,2,3,4,5,6,7), inplace=True)

corr = df.corr()
corr.Survived.sort_values(ascending=False)

from sklearn.model_selection import train_test_split

x = df.drop(['Survived', 'PassengerId'], axis=1)
y = df['Survived']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1)

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy: {}'.format(acc_randomforest))

pickle.dump(randomforest, open('titanic/titanic_model.sav', 'wb'))

df_test = pd.read_csv('test.csv')
df_test['Title'] = df_test['Name'].map(lambda x: get_title(x))

df_test['Title'] = df_test.apply(shorter_titles, axis=1)
ids = df_test['PassengerId']

df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)
df_test['Embarked'].fillna('S', inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)
df_test.drop('Ticket', axis=1, inplace=True)
df_test.drop('PassengerId', axis=1, inplace=True)
df_test.Sex.replace(('male', 'female'), (0,1), inplace=True)
df_test.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)
df_test.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0,1,2,3,4,5,6,7), inplace=True)

predictions = randomforest.predict(df_test)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

def prediction_model(pclass,sex,age,sibsp,parc,fare,embarked,title):
    x = [[pclass,sex,age,sibsp,parc,fare,embarked,title]]
    randomforest = pickle.load(open('titanic/titanic_model.sav', 'rb'))
    predictions = randomforest.predict(x)
    print(predictions)

# prediction_model(1,1,11,1,1,19,1,1)   # just testing

