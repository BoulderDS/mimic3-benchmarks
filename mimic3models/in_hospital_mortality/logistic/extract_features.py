from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import json
import pickle
import csv


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    """
    ret['X'] holds the number of chartevents, each event may not be complete
    """
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def convert_one_hot(v, num, offset=0):
    values = [0 for _ in range(num)]
    values[int(v) + offset] = 1
    return values


def extract_diagnosis_features(file_dir, name):
    pos = name.find("_")
    filename = os.path.join(file_dir, name[:pos], name[pos+1:].replace("_timeseries", ""))
    with open(filename) as fin:
        reader = csv.reader(fin)
        count = 0
        for row in reader:
            count += 1
            features = row
        if count == 2:
            features = convert_one_hot(features[1], 5) + convert_one_hot(features[2], 2, offset=-1) + features[3:6] + features[8:]
            features = [float(v) if v != '' else np.nan for v in features]
        else:
            features = [0] * 138
    return np.array(features)


def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(l2=True)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--all_data', type=str, help='Path to the data of all individual files',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_file', type=str, help='files to store all the features',
                        default='.')
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
    # reorganize data into a dictionary and concatenate with the diagnosis data
    features = {}
    for x, name in zip(train_X, train_names):
        features[name] = np.concatenate((x,
                                        extract_diagnosis_features(os.path.join(args.all_data, "train"), name)))
    for x, name in zip(val_X, val_names):
        features[name] = np.concatenate((x, extract_diagnosis_features(os.path.join(args.all_data, "train"), name)))
    for x, name in zip(test_X, test_names):
        features[name] = np.concatenate((x, extract_diagnosis_features(os.path.join(args.all_data, "test"), name)))
    pickle.dump(features, open(args.output_file, "wb"))

if __name__ == '__main__':
    main()
