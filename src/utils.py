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

def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed



if __name__ == '__main__':
    get_data()
