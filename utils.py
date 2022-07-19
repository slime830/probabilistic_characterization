def count_dict(dictionary, obj):
    if len(dictionary.values()) != 0:
        for value in dictionary.values():
            assert type(value) == int

    if obj in dictionary.keys():
        dictionary[obj] = dictionary.get(obj) + 1
    else:
        dictionary[obj] = 1

    return dictionary


def extend_readlines(filepath, encoding):
    with open(filepath, "r", encoding=encoding) as f:
        lines = [line.replace("\n", "") for line in f.readlines()]

    return lines


def output_strings(str_list, filepath, encoding):
    with open(filepath, "w", encoding=encoding) as f:
        f.writelines(str_list)
