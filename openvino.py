def get_opset(device='cpu'):
    opset = dict()
    idx_val = 2 if device=='gpu' else 1

    with open('openvino.txt', 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        splits = line.strip().split('\t')
        if len(splits) != 3:
            continue

        key = splits[0]
        val = splits[idx_val] == 'Yes'

        opset[key] = val

    return opset
