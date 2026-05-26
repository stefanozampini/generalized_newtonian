import pickle


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
