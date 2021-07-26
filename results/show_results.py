import sys
import os

import pandas as pd
import numpy as np


def read_accuracies_from_csv(path_to_csv):
    dataframe = pd.read_csv(path_to_csv)

    known_acc = dataframe['known_acc'].tolist()
    unknown_acc = dataframe['unknown_acc'].tolist()
    test_acc = dataframe['test_acc'].tolist()

    return np.array(known_acc), np.array(unknown_acc), np.array(test_acc)


def obtain_mean_std(known_acc, unknown_acc, test_acc):
    mean_kk = np.mean(known_acc)
    std_kk = np.std(known_acc)

    mean_ku = np.mean(unknown_acc)
    std_ku = np.std(unknown_acc)

    mean_test = np.mean(test_acc)
    std_test = np.std(test_acc)

    return mean_kk, std_kk, mean_ku, std_ku, mean_test, std_test


def show_3_scenarios_results(path_to_folder):
    possible_csv_file = ['few_shot_k_1_openness_0.csv', 'few_shot_k_1_openness_50.csv', 'few_shot_k_1_openness_100.csv',
                         'few_shot_k_2_openness_0.csv', 'few_shot_k_2_openness_50.csv', 'few_shot_k_2_openness_100.csv',
                         'few_shot_k_4_openness_0.csv', 'few_shot_k_4_openness_50.csv', 'few_shot_k_4_openness_100.csv']

    for csv_name in possible_csv_file:
        path_to_csv = os.path.join(path_to_folder, csv_name)

        known_acc, unknown_acc, test_acc = read_accuracies_from_csv(path_to_csv)

        mean_kk, std_kk, mean_ku, std_ku, mean_test, std_test = obtain_mean_std(known_acc, unknown_acc, test_acc)

        print(csv_name)
        print('mean known acc: {}'.format(mean_kk))
        print('std known acc: {}'.format(std_kk))
        print('mean unknown acc: {}'.format(mean_ku))
        print('std unknown acc: {}'.format(std_ku))
        print('mean test acc: {}'.format(mean_test))
        print('std test acc: {}'.format(std_test))
        print('weighted acc: {}'.format(0.5*mean_kk + 0.5*mean_ku))
        print('')


if __name__ == '__main__':
    show_3_scenarios_results(sys.argv[1])
