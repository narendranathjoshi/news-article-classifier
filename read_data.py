def read_data(filename):
    text = open(filename).read().split("~~~~~")
    text = [line.strip().split("\n") for line in filter(None, text)]

    return text


def read_labels(filename):
    text = open(filename).read().splitlines()
    return text

if __name__ == "__main__":
    print read_labels("trainingSetLabels.dat")  # testing
