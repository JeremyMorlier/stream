"""Analyze the operators of an ONNX workload.

Reports the total operator count, the distinct operator types (ignoring
attributes/shapes), the distinct operators (op_type + attributes + input/output
shapes), and the prime factorization of every dimension of every operator
operand (input/output tensor).
"""

from collections import Counter

import onnx
from onnx import shape_inference

WORKLOAD_PATH = "stream/inputs/examples/workload/resnet18.onnx"


def prime_factors(n):
    """Return the prime factorization of n as a dict {prime: exponent}."""
    if not isinstance(n, int) or n <= 0:
        return {}
    factors = {}
    d = 2
    remaining = n
    while d * d <= remaining:
        while remaining % d == 0:
            factors[d] = factors.get(d, 0) + 1
            remaining //= d
        d += 1
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def format_factorization(factors):
    nb = 0
    for p, e in factors.items():
        print(p, e)
        nb += e
    return nb
    return len([p, e in factors.items()])
    if not factors:
        return "1"
    return " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items()))


def tensor_shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.HasField("dim_value"):
            shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            shape.append(d.dim_param)
        else:
            shape.append(None)
    return tuple(shape)


def attr_value(attr):
    """Convert an onnx AttributeProto to a hashable value."""
    if attr.type == onnx.AttributeProto.TENSOR:
        # External tensor data is not loaded; use shape/dtype/name as identity.
        t = attr.t
        return ("TENSOR", t.name, tuple(t.dims), t.data_type)
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value


def node_attrs(node):
    return tuple(sorted((a.name, attr_value(a)) for a in node.attribute))


def load_shapes(model):
    shapes_by_name = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        shapes_by_name[vi.name] = tensor_shape(vi)
    for init in model.graph.initializer:
        shapes_by_name[init.name] = tuple(init.dims)
    return shapes_by_name


def io_shapes(names, shapes_by_name):
    return tuple(shapes_by_name.get(name, "UNKNOWN") for name in names)


def node_signature(node, shapes_by_name):
    in_shapes = io_shapes(node.input, shapes_by_name)
    out_shapes = io_shapes(node.output, shapes_by_name)
    return (node.op_type, node_attrs(node), in_shapes, out_shapes)


def print_operand_factorization(label, shapes):
    for i, shape in enumerate(shapes):
        if not isinstance(shape, tuple):
            print(f"    {label}[{i}] = {shape} -> UNKNOWN SHAPE")
            continue
        parts = []
        for dim in shape:
            if isinstance(dim, int):
                parts.append(f"{dim}=({format_factorization(prime_factors(dim))})")
            else:
                parts.append(str(dim))
        print(f"    {label}[{i}] shape={shape} -> [{', '.join(parts)}]")


def analyze(workload_path):
    model = onnx.load(workload_path, load_external_data=False)
    model = shape_inference.infer_shapes(model)
    shapes_by_name = load_shapes(model)

    op_type_counts = Counter(node.op_type for node in model.graph.node)
    signatures = [node_signature(node, shapes_by_name) for node in model.graph.node]
    signature_counts = Counter(signatures)
    sort_key = lambda item: (item[0][0], str(item[0][1]), str(item[0][2]), str(item[0][3]))

    print(f"Total operators: {len(model.graph.node)}")
    print()

    print(f"Distinct operator types (ignoring attributes/shapes): {len(op_type_counts)}")
    for op_type, count in sorted(op_type_counts.items()):
        print(f"  {op_type}: {count}")
    print()

    print(f"Distinct operators (op_type + attributes + input/output shapes): {len(signature_counts)}")
    for (op_type, attrs, in_shapes, out_shapes), count in sorted(signature_counts.items(), key=sort_key):
        print(f"  {op_type} attrs={dict(attrs)} in={in_shapes} out={out_shapes}  (x{count})")
    print()

    print("Prime decomposition of each operator operand:")
    for (op_type, attrs, in_shapes, out_shapes), count in sorted(signature_counts.items(), key=sort_key):
        print(f"\n{op_type} attrs={dict(attrs)}  (x{count})")
        print_operand_factorization("in", in_shapes)
        print_operand_factorization("out", out_shapes)


if __name__ == "__main__":
    analyze(WORKLOAD_PATH)
