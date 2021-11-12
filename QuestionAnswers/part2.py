from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import SVM as svm


def q2a():
    print('PRIMAL SVM')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    for hyper_c in hyper_cs:
        print(f'HYPER PARAMETER C: {hyper_c}')
        model = svm.PrimalSvmModel(X, y, epochs=100, hyper_c=hyper_c, rate=1e-1, rate_damping=1e-3, random_seed=False)
        error = 1 - model.test(X, y)
        print('TRAIN ERROR: ', error)
        error = 1 - model.test(X_test, y_test)
        print('TEST ERROR: ', error)
        print('')
        print(model.weights)
        print('\n')
        plt.plot(model.J)
    plt.show()


def q3a():
    print('DUAL SVM')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    for hyper_c in hyper_cs:
        print(f'HYPER PARAMETER C: {hyper_c}')
        model = svm.DualSvmModel(X, y, hyper_c=hyper_c)
        print('')
        print('SUPPORT VECTORS:', len(model.sv_X))
        error = 1 - model.test(X, y)
        print('TRAIN ERROR: ', error)
        error = 1 - model.test(X_test, y_test)
        print('TEST ERROR: ', error)
        print('')
        print(model.calc_weight_vector())
        print(model.bias)
        print('\n')


def q3b():
    print('GAUSS SVM')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        # 500/873,
        # 700/873,
    ]

    gammas = [
        0.1,
        # 0.5,
        # 1,
        # 5,
        # 100,
    ]

    for hyper_c in hyper_cs:
        for gamma in gammas:
            print(f'HYPER PARAMETER C: {hyper_c}')
            print(f'HYPER PARAMETER GAMMA: {gamma}')
            model = svm.DualSvmModel(X, y, hyper_c=hyper_c, kernel=svm.GaussianKernel(0.1))
            print('')
            print('SUPPORT VECTORS:', len(model.sv_X))
            error = 1 - model.test(X, y)
            print('TRAIN ERROR: ', error)
            error = 1 - model.test(X_test, y_test)
            print('TEST ERROR: ', error)
            print('')
            print(model.calc_weight_vector())
            print(model.bias)
            print('\n')
        

def load_bank_note_data(csv: str) -> (pd.DataFrame, pd.Series):
    X_cols = ['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy']
    y_col = 'Label'

    train = load_data(csv)
    X = train[X_cols]
    y = encode_vals(train[y_col])
    return X, y


def load_data(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        Path('bank-note', 'bank-note', csv),
        names=['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy', 'Label']
    )


def encode_vals(y: pd.Series) -> pd.Series:
    return y.replace({0: -1})
