from zigzag.datatypes import Constants
from zigzag.workload.layer_node_abc import LayerNodeABC

from stream.node_tensor import NodeTensor
from stream.workload.node import Node


class PadNode(Node, LayerNodeABC):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        padding: tuple[int, ...],
        allow_zero: bool = False,
    ) -> None:
        """Initialize the PadNode

        Args:
            predecessor: The id of this node's parent.
            padding: Padding to add to the input tensor
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="pad",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

        self.allow_zero = allow_zero
        self.padding = padding
        if len(predecessor) == 0 :
            self.input_operand_source = {}
        else :
            self.input_operand_source = {Constants.LAYER_OP_I: predecessor[0]}

    def pad(self, tensor: NodeTensor):
        """Pad the tensor back to the representation needed for producer/consumer."""
        return tensor.pad(self.padding)
