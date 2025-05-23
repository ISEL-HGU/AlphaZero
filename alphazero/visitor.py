"""Visitors for traversing state-action tree.
"""
import tensorflow as tf

from alphazero.node import Node

class Selector():
    """Base selector.
    
    Selector modifies prior probabilities of root node and selects child node. 

    Note:
        Class that extends this class should override `select()`,  
        `modify_priors()`, and `reset()`. 
    """
    def select(self, node: Node) -> int:
        """Calculate action number of the given node's child to visit.  

        Args:
            node (Node): Internal node.

        Returns:
            int: Action number.
        """
        return NotImplementedError(f'class {self.__class__.__name__} did not \
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
        
        Note: 
            This method overrides `select()` of `Selector`.
        """
    
    def modify_priors(self, priors: tf.Tensor) -> tf.Tensor:
        """Modify the given priors by adding Dirichlet noise.
        
        Note:
            This method overrides `modify_priors()` of `Selector`.        
        """
    
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
        Class that extends this class should override `check_terminal()`. 
    
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
        
    def _pre_visit_internal(self, node: Node) -> int:
        """Get an action number of current internal node's child to visit.  
        
        Args: 
            node (Node): Internal node in which this instance is located.
            
        Returns:
            int: Action number.
        """
        return self.select(node)
        
    def _post_visit_internal(self, node: Node, v: float) -> float:
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
        return node.update(v if node.is_root() 
                           else node.get_r() + self._discount_factor * v)
       
    def _visit_leaf(self, node: Node) -> float:
        """Visit current leaf node.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node.
        """
        if self.check_terminal(node):
            return 0
        
        return self._backup(node, self._expand_and_eval(node))
    
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
            s = Agent.factory.create_state(*self._model.encode(node.get_o()))
            node.set_s(s)
            
            ps, v = self._model.estimate(s)
            ps = self.modify_priors(ps)
        else: 
            a = Agent.factory.create_action(node.get_a())
            node.set_a(a)
            
            r, s = self._model.process(node.get_intern_s(), a)
            node.set_r(r)
            node.set_s(s)
            
            ps, v = self._model.estimate(s)
        
        for i, p in enumerate(ps):
            node.add(Node(s, i, p))
        
        return (-2 * self._is_multiagent + 1) * v 
    
    def reset(self) -> None:
        """Reset states of this instance's selector.
        """
        self._selector.reset()
    
    @classmethod
    def set_is_multiagent(cls, is_multiagent) -> None:
        cls._is_multiagent = is_multiagent
    
    def set_model(self, model: Model) -> None:
        self._model = model        
