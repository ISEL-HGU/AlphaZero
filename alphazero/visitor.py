"""Visitors for traversing state-action tree.
"""

from alphazero.forwardables.forwardable import Forwardable
from alphazero.mdp import MDPFactory
from alphazero.node import Node

class NodeVisitor():
    """Base node visitor.  
      
    Each node visitor conducts different action selection algorithm when  
    traversing the tree. Class that inherits this class should override  
    `select()`. 
    
    Attributes:
        _forwardable (Forwardable): Simulator or neural network.
        _factory (MDPFactory): MDP factory.
        _gamma (float): Discount factor.
        _is_multiagent (bool): Value of multiagent perspective.
    """
    
    def __init__(self, forwardable: Forwardable, factory: MDPFactory, 
                 gamma: float, multiagent: bool):
        """Initialize `NodeVisitor` instance.

        Args:
            forwardable (Forwardable): Simulator or neural network that can   
                process one step.
            factory (MDPFactory): MDP factory.
            gamma (float): Discount factor
            multiagent (bool): Value that indicates whether the tree   
                traversing is conducted in multiagent perspective or not.
        """
        self._forwardable = forwardable
        self._factory = factory
        self._gamma = gamma 
        self._multiagent = multiagent
    
    def _pre_visit_internal(self, node: Node) -> int:
        """Get action number for visiting child of the current internal node.  
        
        Args: 
            node (Node): Internal node in which this instance is located.
            
        Returns:
            int: Action number.
        """
    
    def select(self, node: Node) -> int:
        """Calculate action number of the current internal node's child.  

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
        """Update statistics of current node.

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
    
    def _expand_and_eval(self, node: Node) -> float:
        """Expand the children and evaluate state value of current leaf node.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node.
        """