from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import SVM as svm


def q2a():
    print('PRIMAL SVM: 2a')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    results_cols = X.columns.to_list()
    results_cols.append('MODEL_BIAS')
    results_cols.append('TrainingError')
    results_cols.append('TestError')
    results_cols.append('C')

    results = pd.DataFrame(columns=results_cols)
    print(results)

    for i, hyper_c in enumerate(hyper_cs):
        print(f'HYPER PARAMETER C: {hyper_c}')
        model = svm.PrimalSvmModel(X, y, epochs=100, hyper_c=hyper_c, rate=1e-1, rate_damping=1e-3, random_seed=False)
        train_error = 1 - model.test(X, y)
        print('TRAIN ERROR: ', train_error)
        test_error = 1 - model.test(X_test, y_test)
        print('TEST ERROR: ', test_error)
        print('')
        print(model.weights)
        print('\n')
        plt.plot(model.J)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.savefig(Path('Instructions', f'part2_q2a_cost_plot_{i}.png'))
        plt.close()

        result = dict()
        for i, x in enumerate(X.columns):
            result[x] = model.weights[i]
        result['TrainingError'] = train_error
        result['TestError'] = test_error
        result['C'] = hyper_c
        results = results.append(pd.Series(result), ignore_index=True).reset_index(drop=True)

    print(results)

    results.to_csv('part2_q2a.csv', index=False)


def q2b():
    print('PRIMAL SVM: 2b')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    results_cols = X.columns.to_list()
    results_cols.append('MODEL_BIAS')
    results_cols.append('TrainingError')
    results_cols.append('TestError')
    results_cols.append('C')

    results = pd.DataFrame(columns=results_cols)
    print(results)

    for i, hyper_c in enumerate(hyper_cs):
        print(f'HYPER PARAMETER C: {hyper_c}')
        model = svm.PrimalSvmModel(X, y, epochs=100, hyper_c=hyper_c, rate=1, rate_damping=1, random_seed=False)
        train_error = 1 - model.test(X, y)
        print('TRAIN ERROR: ', train_error)
        test_error = 1 - model.test(X_test, y_test)
        print('TEST ERROR: ', test_error)
        print('')
        print(model.weights)
        print('\n')
        plt.plot(model.J)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.savefig(Path('Instructions', f'part2_q2b_cost_plot_{i}.png'))
        plt.close()

        result = dict()
        for i, x in enumerate(X.columns):
            result[x] = model.weights[i]
        result['TrainingError'] = train_error
        result['TestError'] = test_error
        result['C'] = hyper_c
        results = results.append(pd.Series(result), ignore_index=True).reset_index(drop=True)

    print(results)

    results.to_csv('part2_q2b.csv', index=False)


def q3a():
    print('DUAL SVM')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    results_cols = X.columns.to_list()
    results_cols.append('MODEL_BIAS')
    results_cols.append('TrainingError')
    results_cols.append('TestError')
    results_cols.append('C')

    results = pd.DataFrame(columns=results_cols)
    print(results)

    for hyper_c in hyper_cs:
        print(f'HYPER PARAMETER C: {hyper_c}')
        model = svm.DualSvmModel(X, y, hyper_c=hyper_c)
        print('')
        print('SUPPORT VECTORS:', len(model.sv_X))
        train_error = 1 - model.test(X, y)
        print('TRAIN ERROR: ', train_error)
        test_error = 1 - model.test(X_test, y_test)
        print('TEST ERROR: ', test_error)
        print('')
        print(model.calc_weight_vector())
        print(model.bias)
        print('\n')

        w = model.calc_weight_vector()
        result = dict()
        for i, x in enumerate(X.columns):
            result[x] = w[i]
        result['MODEL_BIAS'] = model.bias
        result['TrainingError'] = train_error
        result['TestError'] = test_error
        result['C'] = hyper_c
        results = results.append(pd.Series(result), ignore_index=True).reset_index(drop=True)

        print(results)

    results.to_csv('part2_q3a.csv')


def q3b_c():
    print('GAUSS SVM')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    hyper_cs = [
        100/873,
        500/873,
        700/873,
    ]

    gammas = [
        0.1,
        0.5,
        1,
        5,
        100,
    ]

    b_results_cols = [
        'C',
        'GAMMA',
        'TrainingError',
        'TestError',
    ]

    b_results = pd.DataFrame(columns=b_results_cols)
    print(b_results)

    c_results_list = list()

    for hyper_c in hyper_cs:
        for gamma in gammas:
            print(f'HYPER PARAMETER C: {hyper_c}')
            print(f'HYPER PARAMETER GAMMA: {gamma}')
            model = svm.DualSvmModel(X, y, hyper_c=hyper_c, kernel=svm.GaussianKernel(gamma))
            print('')
            print('SUPPORT VECTORS:', len(model.sv_X))
            train_error = 1 - model.test(X, y)
            print('TRAIN ERROR: ', train_error)
            test_error = 1 - model.test(X_test, y_test)
            print('TEST ERROR: ', test_error)
            print('')
            print(model.calc_weight_vector())
            print(model.bias)
            print('\n')

            w = model.calc_weight_vector()
            result = dict()
            result['GAMMA'] = gamma
            result['TrainingError'] = train_error
            result['TestError'] = test_error
            result['C'] = hyper_c
            b_results = b_results.append(pd.Series(result), ignore_index=True).reset_index(drop=True)
            print(b_results)

            if hyper_c == 500 / 873:
                c_results_list.append(model.sv_ix)

    c_results = pd.DataFrame()
    for (i, a), (j, b) in combinations(enumerate(c_results_list), 2):
        len_overlap = len(np.intersect1d(a, b))

        c_results = c_results.append(
            {'gamma_i': gammas[i],'gamma_j': gammas[j],'len_overlap': len_overlap,}, 
            ignore_index=True
        )

    b_results.to_csv('part2_q3b.csv')
    c_results.to_csv('part2_q3c.csv')

    print('Part C')
    print(c_results)
        

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
