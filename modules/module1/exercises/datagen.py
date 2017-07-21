import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = '/dsa/data/all_datasets/titanic3.csv'
DATASET_TRAIN = './datasets/titanic.csv'
DATASET_TEST = './datasets/titanic.test.csv'
DATASET_KEY = './datasets/titanic.key.csv'
assert os.path.exists(DATASET)

def load():
    dataset = pd.read_csv(DATASET)
    dataset = dataset.loc[:, [
        'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]

    dataset.embarked.fillna("S", inplace=True)
    dataset.embarked = list(map(['C', 'S', 'Q'].index, dataset.embarked))
    dataset.sex = list(map(['female', 'male'].index, dataset.sex))
    dataset.fare.fillna(dataset.fare.median(), inplace=True)
    np.random.seed(1234)

    age_mean = dataset.age.mean()
    age_std = dataset.age.std()
    age_na = dataset.age.isnull()
    _age = np.array(dataset.age)
    _age[age_na] = np.random.randint(age_mean - age_std, age_mean + age_std, size = np.sum(age_na))
    dataset.age = _age

    assert np.all(np.logical_not(dataset.isnull()))

    return dataset

def save(dataset):
    os.system('mkdir -p datasets')
    X = dataset.iloc[:, :-1]
    Y = dataset.survived
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.68, random_state = 1234)
    pd.concat([X_train, Y_train], axis = 1).to_csv(DATASET_TRAIN, index = False)
    X_test.to_csv(DATASET_TEST, index = False)
    Y_test.to_csv(DATASET_KEY, index = False)

def main():
    save(load())

if __name__ == '__main__':
    main()
