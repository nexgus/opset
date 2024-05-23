import json
import onnx
import os


def opset_version(model):
    return model.opset_import[0].version


def record(d, key, val):
    if key in d:
        d[key].append(val)
    else:
        d[key] = [val]


def to_textproto(model, filepath):
    dirname = os.path.dirname(filepath)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)

    import google.protobuf.text_format
    with open(filepath, 'w') as fp:
        fp.write(google.protobuf.text_format.MessageToString(model))


def validate_opset(model, device, filepath):
    print(f'producer_name: {model.producer_name}')
    print(f'producer_version: {model.producer_version}')

    ver = opset_version(model)
    print(f'opset version: {ver}')
    if ver > 12:
        raise ValueError(f'Unsupported ONNX opset version: {ver}')

    if device == 'kl720':
        from kl720 import opset
    else:
        raise ValueError(f'Unsupported NPU device: {device}')

    supported = dict()
    not_supported = dict()
    unknown = dict()
    for node in model.graph.node:
        name = node.name
        op_type = node.op_type
        try:
            if opset[op_type]:
                record(supported, op_type, name)
            else:
                record(not_supported, op_type, name)
        except KeyError:
            record(unknown, op_type, name)

    dirname = os.path.dirname(filepath)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)

    with open(filepath, 'w') as fp:
        json.dump({
            'not_supported': not_supported,
            'supported': supported,
            'unknown': unknown,
        }, fp, indent=4)


def main(args):
    model = onnx.load(args.model)
    if args.subcmd == 'list':
        raise NotImplementedError
    elif args.subcmd == 'textproto':
        if args.out is None:
            args.out = os.path.splitext(os.path.basename(args.model))[0] + '.textproto'
        to_textproto(model, args.out)
    elif args.subcmd == 'validate':
        if args.out is None:
            args.out = os.path.splitext(os.path.basename(args.model))[0] + '.json'
        validate_opset(model, args.device, args.out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Scan ONNX opset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('model', type=str, help='Path to the ONNX model')

    subcmd = parser.add_subparsers(dest='subcmd', help='subcommands', metavar='SUBCOMMAND')
    subcmd.required = True

    list_parser = subcmd.add_parser('list', help='List all used ops')

    txtproto_parser = subcmd.add_parser('textproto', help='Convert to textproto')
    txtproto_parser.add_argument('--out', '-o', type=str, help='Path to the output file')

    validate_parser = subcmd.add_parser('validate', help='Validate opset')
    validate_parser.add_argument('--device', '-d', default='kl720', choices=['kl720'], help='The NPU device')
    validate_parser.add_argument('--out', '-o', type=str, help='Path to the output file')

    args = parser.parse_args()
    main(args)
