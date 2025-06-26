"""Node visitors and selectors."""

from math import sqrt, log
from tkinter import N
import tensorflow as tf
import tensorflow_probability as tfp

from alphazero.alphazero import BaseAlphaZero
from alphazero.components.agent import Agent
from alphazero.nn.model import Model
from alphazero.tree.node import Node

class Selector():
    """Base selector.
    
    Selector modifies prior probabilities of root node and selects child node. 

    Note:
        Descendants of `Selector` should override `select()`,  
        `modify_priors()`, and `reset()`. 
        
    Attributes: 
        _min_q (float): Minimum state-action value of the search tree.
        _max_q (float): Maximum state-action value of the search tree.
    """
    
    def __init__(self):
        """Initialize `Selector` instance.
        """
        self._min_q = float('-inf')
        self._max_q = float('inf')
    
    def update_min_max_q(self, q: float) -> None:
        """Update `min_q` or `max_q` with the given state-action value.  

        Args:
            q (float): The state-action value.
        """
        if q < self._min_q:
            self._min_q = q
        elif q > self._max_q:
            self._max_q = q
    
    def select(self, node: Node) -> int:
        """Calculate action number of the given node's child to visit.  

        Args:
            node (Node): Internal node.

        Returns:
            int: Action number.
        """
        raise NotImplementedError(f'class {self.__class__.__name__} did not \
                                  implement select().')
    
    def modify_priors(self, priors: tf.Tensor) -> tf.Tensor:
        """Modify the given priors by adding noise.
        
        Args:
            priors (tf.Tensor): Prior probabilities.

        Returns:
            tf.Tensor: Prior probabilities with noise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  modify_priors().')
    
    def reset(self) -> None:
        """Reset states of this instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  reset().')

class PUCTSelector(Selector):
    """Selector that modifies prior probabilities with Dirichlet noise and  
    selects child node with PUCT algorithm.

    Attributes:
        _diric_param (float): Dirichlet parameter.
        _diric_weight (float): Dirichlet noise weight.
    """
    def __init__(self, diric_param: float, diric_weight: float):
        """Initialize `PUCTSelector` instance.

        Args:
            diric_param (float): Dirichlet parameter.
            diric_weight (float): Dirichlet noise weight.
        """
        self._diric_param = diric_param
        self._diric_weight = diric_weight

    def select(self, node: Node) -> int:
        """Calculate action number of given node's child to visit with PUCT  
        algorithm.
        
        The formula for caculating the action number is as following:  
        
        TO DO:
            Add PUCT formula to this docstring.
        
        Note: 
            This method overrides `select()` of `Selector`.
        """
        C1 = 1.25
        C2 = 19652
        
        return tf.argmax(tf.constant(
                [child.get_q() + child.get_p() 
                               * sqrt(node.get_n()) 
                               / (1 + child.get_n()) 
                               * (C1 + log((node.get_n() + C2 + 1) / C2)) 
                 for child in node.get_children()])).numpy()
        
    def modify_priors(self, priors: tf.Tensor) -> tf.Tensor:
        """Modify the given priors by adding Dirichlet noise.
        
        Note:
            This method overrides `modify_priors()` of `Selector`.        
        """
        #tfp.distributions.Dirichlet(
        #        tf.fill(priors.shape.as_list(), self._diric_param)).sample    
    
    def reset(self) -> None:
        """Reset states of this instance.
        
        This method does nothing since there are no states in this instance.
        
        Note:
            Override `reset()` of `Selector`.
        """
        pass 

class NodeVisitor():
    """Base node visitor.  
    
    Node visitor conducts one simulation of mcts when traversing the tree.  
    
    Note:
        Descendants of `NodeVisitor` should override `check_terminal()`.

    Attributes:
        _model (Model): Neural network model.
         _selector (Selector): Selector.
        _discount_factor (float): Discount factor.
    """
    
    def __init__(self, model: Model, selector: Selector, 
                 discount_factor: float):
        """Initialize `NodeVisitor` instance.

        Args:
            model (Model): Neural network model.
            selector (Selector): Selector.
            discount_factor (float): Discount factor.
        """
        self._model = model
        self._selector = selector
        self._discount_factor = discount_factor 
        
    def pre_visit_internal(self, node: Node) -> int:
        """Get an action number of current internal node's child to visit.  
        
        Args: 
            node (Node): Internal node in which this instance is located.
            
        Returns:
            int: Action number.
        """
        return self._selector.select(node)
        
    def post_visit_internal(self, node: Node, v: float) -> float:
        """Back up current internal node with its visited child's state value.

        Args:
            node (Node): Internal node in which this instance is located.
            v (float): State value of the current internal node's child.

        Returns:
            float: State value of the current internal node.
        """
        return self._backup(node, v)
    
    def _backup(self, node: Node, v: float) -> float: 
        """Update statistics of the given node.

        Args:
            node (Node): Node in which this instance is located.
            v (float): State value of the current node's child.

        Returns:
            float: State value of the current node.
        """
        if node.is_root():
            return node.update_stats(v)
        else:
            g = node.get_r() + self._discount_factor * v
            
            self._selector.update_min_max_q(g)
            
            return node.update_stats(g)
       
    def visit_leaf(self, node: Node) -> float:
        """Visit current leaf node.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node.
        """
        return self._backup(node, 
                            0 if self.check_terminal(node) 
                            else self._expand_and_eval(node))
    
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
        if node.is_root():
            s = BaseAlphaZero.factory.create_state(*self._model.encode(node.get_o()))
            node.set_s(s)
            
            ps, v = self._model.estimate(s)
            ps = self._selector.modify_priors(ps)
        else: 
            r, s = self._model.process(node.get_intern_s(), node.get_a())
            node.set_r(r)
            node.set_s(s)
            
            if s is None or s.is_terminal():
                return 0
             
            ps, v = self._model.estimate(s)
        
        for i, p in enumerate(ps):
            node.add(Node(s, BaseAlphaZero.factory.create_action(i), p))
        
        return (-2 * self._is_multiagent + 1) * v 
    
    def reset_selector(self) -> None:
        """Reset `selector` of this instance.
        """
        self._selector.reset()
    
    @classmethod
    def set_is_multiagent(cls, is_multiagent) -> None:
        cls._is_multiagent = is_multiagent
    
    def set_model(self, model: Model) -> None:
        self._model = model        
