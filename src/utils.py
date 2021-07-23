def get_classes_list(classes_file):
    classes = []
    with open(classes_file) as f:
        classes = [line.rstrip() for line in f]

    return classes


def get_data(data_file):
    data = []
    with open(data_file) as f:
        data = [line.rstrip() for line in f]
        data = [x for x in data if x]

    res = {}
    for d in data:
        key, val = d.split('=')
        res[key.strip()] = val.strip()

    return res


if __name__ == '__main__':
    get_data()
