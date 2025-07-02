from typing import Any

from zigzag.datatypes import Constants

from stream.workload.computation.computation_node import ComputationNode
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.parser.onnx.utils import get_onnx_tensor_type, get_attribute_ints_with_name
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser

from stream.parser.onnx.reduce_1d import Reduce1DParser
from stream.parser.onnx.conv import ConvParser
from stream.parser.onnx.convtranspose import ConvTransposeParser

from stream.parser.onnx.transpose import TransposeParser
from stream.workload.dependency_propagation.transpose_node import TransposeNode
class ConvGradParser(OnnxComputeOperatorParser):

    NODE_TYPES = ["conv", "conv", "sum", "transpose", "transpose", "transpose"]

    def run(self):
        for node in self.get_nodes():
            yield node

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...

    def get_nodes(self):
        # Parse initial CNs
        self.parse_into_subnodes()
        # Give correct op type and name
        self.set_nodes_name_and_type()
        # Override dependencies
        self.correct_nodes_operand_source()

        return self.nodes
    
    def parse_into_subnodes(self):
        """Prase the base ONNX node multiple times into the different Computation Nodes.
        The CNs that result from this operation have some incorrect properties regarding the graph structure
        """
        # parser_classes: list[type] = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser, SoftmaxCrossEntropySIMDParser, Reduce1DParser, SoftmaxCrossEntropyReduceParser]
        parser_classes: list[type] = [ConvGradWeightParser, ConvGradOutputParser, ConvGradBiasParser, TransposeNodeWeightGrad1, TransposeNodeWeightGrad2, TransposeNodeWeightGrad3]
        node_ids = [self.node_id + i for i in range(3)]
        parsers: list[OnnxComputeOperatorParser] = [
            parser(
                node_id=node_id,
                node=self.node,
                nodes_outputs=self.nodes_outputs,
                onnx_model=self.onnx_model,
                all_mappings=self.all_mappings,
                accelerator=self.accelerator,
            )
            for parser, node_id in zip(parser_classes, node_ids)
        ]
        self.nodes = []
        for parser in parsers :
            for node in parser.run() :
                self.nodes.append(node)
        self.nodes = tuple(self.nodes)

    def set_nodes_name_and_type(self):
        """Set the name and operator type of all Computation Nodes that stem from the base ONNX node"""
        for node, node_type in zip(self.nodes, ConvGradParser.NODE_TYPES):
            node.type = node_type
            node.name += f"-{node_type}/"

    def correct_nodes_operand_source(self):
        """Correct the `input_operand_source` and `constant_operands` of all Computation Nodes that stem from the base
        ONNX node"""
        op_I = Constants.LAYER_OP_I
        op_W = Constants.LAYER_OP_W

        node_grad_weight, node_grad_output, node_grad_bias, node_transpose1, node_transpose2, node_transpose3 = self.nodes
        id_grad_weight, id_grad_output, id_grad_bias, id_transpose1, id_transpose2, id_transpose3 = [node.id for node in self.nodes]
        prev_node_id = node_grad_weight.input_operand_source[op_I]
    
        # fix nodes inputs
        node_grad_weight.input_operand_source = {}
        node_grad_output.input_operand_source = {}
        node_grad_bias.input_operand_source = {}
        # node_transpose1.input_operand_source = {op_I:}
        # node_transpose2.input_operand_source = {op_I:}
        node_transpose3.input_operand_source = {op_I:id_grad_weight}

        # node_sfm_max, node_sfm_exp, node_sfm_sum, node_sfm_div, node_log, node_sum, node_reduce = self.nodes
        node_sfm_max, node_sfm_exp, node_sfm_sum, node_sfm_div, node_log = self.nodes
        # id_sfm_max, id_sfm_exp, id_sfm_sum, id_sfm_div, id_log, id_sum, id_reduce = [node.id for node in self.nodes]
        id_sfm_max, id_sfm_exp, id_sfm_sum, id_sfm_div, id_log = [node.id for node in self.nodes]
        prev_node_id = node_sfm_max.input_operand_source[op_I]  # Node before Softmax

        # Default after generation: input_operand_source = {op_I: prev_node_id} and constant_operands = [W]
        node_sfm_exp.input_operand_source = {op_I: prev_node_id, op_W: id_sfm_max}
        node_sfm_exp.constant_operands = []
        node_sfm_sum.input_operand_source = {op_I: id_sfm_exp}
        node_sfm_div.input_operand_source = {op_I: id_sfm_exp, op_W: id_sfm_sum}
        node_sfm_div.constant_operands = []

        # TODO: fix constant operands for added nodes
        node_log.input_operand_source = {op_I: id_sfm_div}
        # node_sum.input_operand_source = {op_I: id_log}
        # node_reduce.input_operand_source = {op_I: id_sum}
        node_log.constant_operands = []
        # node_sum.constant_operands = []
        # node_reduce.constant_operands = []

class TransposeNodeWeightGrad1(TransposeNode) :
    def generate_node(self):
        predecessors = self.get_node_predecessors()
        predecessor = predecessors.pop()

        permute_axes = [1, 0, 2 , 3]

        return TransposeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            permute_axes=permute_axes,
        )

class TransposeNodeWeightGrad2(TransposeNode) :
    def generate_node(self):
        predecessors = self.get_node_predecessors()
        predecessor = predecessors.pop()

        permute_axes = [1, 0, 2 , 3]

        return TransposeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            permute_axes=permute_axes,
        )
    
class TransposeNodeWeightGrad3(TransposeNode) :
    def generate_node(self):
        predecessors = self.get_node_predecessors()
        predecessor = predecessors.pop()

        permute_axes = [1, 0, 2 , 3]

        return TransposeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            permute_axes=permute_axes,
        )
class ConvGradWeightParser(ConvParser) :
    def generate_node(self):
        attrs = self.node.attribute
        # dilations and strides are inverted
        dilations: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
        strides: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore

        #TODO: check for padding and group size difference
        group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore

        # Get the input and output activation shapes
        grad_shape, activations_shape, weight_shape, output_grad_shape = convgrad_get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data: dict[str, Any] = self.get_layer_node_user_format(
            grad_shape,
            strides,
            dilations,
            group_size,
            padding,
            activations_shape,
            weight_shape,
        )

        node_factory = LayerNodeFactory(node_data, mapping_data=None)
        node_attrs = node_factory.create_node_attr()
        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            mapping_attr=mapping,
            op_type=ConvParser.OP_TYPE,
            operand_tensor_reshape=None,
        )

class ConvGradOutputParser(ConvTransposeParser) :  
    def generate_node(self):
        attrs = self.node.attribute
        kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
        strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
        dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
        group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore

        # Get the input and output activation shapes
        grad_shape, activations_shape, weight_shape, output_grad_shape = convgrad_get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        node_data: dict[str, Any] = self.get_layer_node_user_format(
            kernel_shape,
            strides,
            dilations,
            group_size,
            padding,
            grad_shape,
            activations_shape,
        )

        node_factory = LayerNodeFactory(node_data, mapping_data=None)
        node_attrs = node_factory.create_node_attr()
        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            mapping_attr=mapping,
            op_type=ConvParser.OP_TYPE,
            operand_tensor_reshape=None,
        )
    
class ConvGradBiasParser(Reduce1DParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        grad_shape, activations_shape, weight_shape, output_grad_shape = convgrad_get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # From the ONNX node
        node_data = self.get_layer_node_user_format(grad_shape, None)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )
def convgrad_get_node_input_output_dimension_shapes(node, model) :
    # weight_grad and bias_grad may not be known 

    grad = node.input[0]
    grad_shape = get_onnx_tensor_type(grad, model).shape
    activations = node.input[1]
    activations_shape = get_onnx_tensor_type(activations, model).shape
    weight = node.input[2]
    weight_shape = get_onnx_tensor_type(weight, model).shape

    output_grad = node.output[0]

    output_grad_shape = get_onnx_tensor_type(output_grad, model).shape
    return grad_shape, activations_shape, weight_shape, output_grad_shape
