"""Convert the DeblurGAN-v2 Keras generator to ONNX for in-browser inference.

Weights are stored as float16 and cast back to float32 at load time, which halves the
download while keeping float32 compute on every ONNX Runtime Web backend.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CACHE = Path(os.getenv("AI_IMAGE_MODEL_CACHE", Path.home() / ".cache" / "ai-image-enhancement")).expanduser()


def convert(tile: int = 256) -> onnx.ModelProto:
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(CACHE / "deblurgan_generator.keras", compile=False)
    spec = [tf.TensorSpec([1, tile, tile, 3], tf.float32, name="input")]

    @tf.function(input_signature=spec)
    def forward(x):
        return model(x, training=False)

    proto, _ = tf2onnx.convert.from_function(forward, input_signature=spec, opset=17)
    return proto


def shrink(proto: onnx.ModelProto) -> onnx.ModelProto:
    graph = proto.graph
    new_nodes = []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < 64:
            continue
        values = numpy_helper.to_array(init)
        name = init.name
        half_name = name + "__fp16"
        graph.initializer.remove(init)
        graph.initializer.append(numpy_helper.from_array(values.astype(np.float16), half_name))
        new_nodes.append(helper.make_node("Cast", [half_name], [name], to=TensorProto.FLOAT, name=name + "__cast"))
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes + nodes)
    return proto


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-onnx", type=Path, help="Shrink an already converted fp32 ONNX file instead of converting the Keras model.")
    args = parser.parse_args()
    result = shrink(onnx.load(args.from_onnx) if args.from_onnx else convert())
    onnx.checker.check_model(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
