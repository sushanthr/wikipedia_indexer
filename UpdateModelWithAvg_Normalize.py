import onnx
from onnx import helper
import numpy as np

def add_average_pool_and_normalize(input_model_path, output_model_path):
    # Load the ONNX model
    model = onnx.load(input_model_path)

    # Get the graph
    graph = model.graph

    # Get the output of the last node
    last_output = graph.output[0]

    # Create new nodes for average pooling and normalization
    # Average pooling
    reduce_mean_node = helper.make_node(
        'ReduceMean',
        inputs=[last_output.name],
        outputs=['pooled_output'],
        axes=[1],  # Average along sequence dimension
        keepdims=0
    )

    # L2 Normalization
    l2_norm_node = helper.make_node(
        'LpNormalization',
        inputs=['pooled_output'],
        outputs=['normalized_output'],
        axis=1,
        p=2
    )

    # Add new nodes to the graph
    graph.node.extend([reduce_mean_node, l2_norm_node])

    # Update the graph output
    new_output = helper.make_tensor_value_info('normalized_output', onnx.TensorProto.FLOAT, [None, None])
    graph.output[0].CopyFrom(new_output)

    # Save the modified model
    onnx.save(model, output_model_path)

# Usage
input_model_path = "D:\\Projects\\e5-small-v2\\model_opt1_QInt8.onnx"
output_model_path = "D:\\Projects\\e5-small-v2\\model_with_pooling_and_norm.onnx"
add_average_pool_and_normalize(input_model_path, output_model_path)