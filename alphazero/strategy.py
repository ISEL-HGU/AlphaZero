"""
"""
import tensorflow as tf

from alphazero.node import Node

class PolicyImprovementStrategy():
    """Base policy improvement strategy class.
    
    Class that extends this class should override `improve()`.
    """
    
    def improve(self, node: Node) -> tf.Tensor:
        """Improve policy using statistics of the given node.

        Args:
            node (Node): Root node.
        
        Returns: 
            tf.Tensor: Improved policy.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  improve().')