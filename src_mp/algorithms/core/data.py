from src.algorithms.utils.input import normalize_dataset
from src.algorithms.utils.load_data import load_data
from src.data_gen import prepare_data


def load_federated_data(args, pfl: bool = False, normalize: bool = True):
    prepare_data(args)
    train_sets, test_set, train_counts, num_class = load_data(args, pfl=pfl)
    if normalize:
        for dataset in train_sets.values():
            normalize_dataset(dataset, args.dataset)
        if pfl:
            for dataset in test_set.values():
                normalize_dataset(dataset, args.dataset)
        else:
            normalize_dataset(test_set, args.dataset)
    return train_sets, test_set, train_counts, num_class
