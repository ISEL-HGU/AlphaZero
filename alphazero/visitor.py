"""Visitors for traversing state-action tree.
"""
import tensorflow as tf

from alphazero.stepables.stepable import Stepable
from alphazero.mdp import MDPFactory
from alphazero.node import Node

class NodeVisitor():
    """Base node visitor.  
      
    Each node visitor conducts different action selection algorithm when  
    traversing the tree. Class that inherits this class should override  
    `select()`. 
    
    Attributes:
        _model (AlphaZeroModel): AlphaZero model.
        _factory (MDPFactory): MDP factory.
        _discount_factor (float): Discount factor.
        _is_multiagent (bool): Value of multiagent perspective.
    """
    
    def __init__(self, model: AlphaZeroModel, factory: MDPFactory, 
                 discount_factor: float, multiagent: bool):
        """Initialize `NodeVisitor` instance.

        Args:
            model (AlphaZeroModel): AlphaZero model.
            factory (MDPFactory): MDP factory.
            discount_factor (float): Discount factor
            multiagent (bool): Value that indicates whether the tree   
                traversing is conducted in multiagent perspective or not.
        """
        self._model = model
        self._factory = factory
        self._discount_factor = discount_factor 
        self._multiagent = multiagent
    
    def _pre_visit_internal(self, node: Node) -> int:
        """Get an action number of current internal node's child to visit.  
        
        Args: 
            node (Node): Internal node in which this instance is located.
            
        Returns:
            int: Action number.
        """
    
    def select(self, node: Node) -> int:
        """Calculate action number of the given node's child to visit.  

        Args:
            node (Node): Internal node in which this instance is located.

        Returns:
            int: Action number.
        """
        return NotImplementedError(f'class {self.__class__.__name__} did not \
                                   implement select().')
        
    def _post_visit_internal(self, node: Node, v: float) -> float:
        """Back up current internal node with its visited child's state value.

        Args:
            node (Node): Internal node in which this instance is located.
            v (float): State value of the current internal node's child.

        Returns:
            float: State value of the current internal node.
        """
    
    def _backup(self, node: Node, v: float) -> float: 
        """Update statistics of the given node.

        Args:
            node (Node): Node in which this instance is located.
            v (float): State value of the current node's child.

        Returns:
            float: State value of the current node.
        """
    
    def _visit_leaf(self, node: Node) -> float:
        """Visit current leaf node.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node.
        """
    
    def check_terminal(self, node: Node) -> bool:
        """Check whether the given node is terminal or not.

        Args:
            node (Node): Node in which this instance is located.

        Returns:
            bool: `True` if the given node is terminal node, `False` otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  check_terminal().')
    
    def _expand_and_eval(self, node: Node) -> float:
        """Expand the given node and evaluate the state value of the given  
        node's children.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node's children.
        """
    
    def modify_priors(self, priors: tf.Tensor) -> list:
        """Modify the given priors by adding noise.

        Args:
            priors (tf.Tensor): Prior probabilities.

        Returns:
            list: Prior probabilities with noise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  modify_priors().')
