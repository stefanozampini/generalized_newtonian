import pickle


def pickle_save(filename, d):
    with open(filename, "wb") as f:
        pickle.dump(d, f)


def pickle_load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except:
        return None
