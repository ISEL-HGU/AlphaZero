"""Node visitors and selectors."""

import tensorflow as tf
import tensorflow_probability as tfp

from alphazero.alphazero import BaseAlphaZero
from alphazero.nn.model import Model
from alphazero.tree.node import Node
from alphazero.utils.math import puct


class Selector():
    """Base selector.

    Note:
        Descendants of `Selector` should override `select()` and `reset()`.  
        They also should call `__init__` of `Selector` in the constructor. 
        
    Attributes:
        _min_q (float): Minimum state-action value of the search tree.
        _max_q (float): Maximum state-action value of the search tree.
    """
    
    def __init__(self):
        """Initialize `Selector` instance.
        """
        self._min_q = float('-inf')
        self._max_q = float('inf')
    
    def update_min_max_q(self, q: float) -> float:
        """Update minimum state-action value or maximum state-action value  
        of the search tree with given state-action value.  

        Args:
            q (float): The state-action value.
            
        Returns:
            float: The state-action value.
        """
        if q < self._min_q:
            self._min_q = q
        elif q > self._max_q:
            self._max_q = q
        
        return q
    
    def select(self, node: Node) -> int:
        """Calculate action number of the given node's child to visit.  

        Args:
            node (Node): Internal node.

        Returns:
            int: Action number.
        """
        raise NotImplementedError(f'class {self.__class__.__name__} did not \
                                  implement select().')
    
    def reset(self) -> None:
        """Reset states of this instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  reset().')


class PUCTSelector(Selector):
    """Selector that modifies prior probabilities with Dirichlet noise and  
    selects child node with PUCT algorithm.

    Attributes:
        _noise (tf.Tensor): Diriclet noise
        _concentration (float): Concenctration parameter.
        _weight (float): Noise weight.
    """
    def __init__(self, concentration: float, weight: float, n):
        """Initialize `PUCTSelector` instance.

        Args:
            concentration (float): Concentration parameter of dirichlet  
                distribution.
            weight (float): Dirichlet noise weight.
        """
        super(PUCTSelector, self).__init__()
        
        self._noise = None
        self._concentration = concentration
        self._weight = weight

    def select(self, node: Node) -> int:
        """Calculate action number of the child node to visit with PUCT  
        algorithm.
        
        The formula for caculating the action number is as following:
        .. math::
            a=\\underset{a}{\\mathrm{argmax}}\\Bigg[Q(s,a)+P(s,a)\\frac{ \\
            \\sqrt{\\sum_{b}N(s,b)}}{1+N(s,a)}\\Bigg(c_1+\\log(\\frac{ \\
            \\sum_{b}N(s,b)+c_2+1}{c_2}))],
        
        where :math:`\\sum_{b}N(s,b)` denotes visit count of the parent node.  
        :math:`c_1=1.25` and :math:`c_2=19652` is used. The formula is  
        referenced by Schrittwieser et al., 2020.
        
        Note: 
            This method overrides `select()` of `Selector`.
        
        See Also:
            https://www.nature.com/articles/s41586-020-03051-4
        """
        stats = node.obtain_stats_children()
        
        if self._noise is None: 
            self._noise = tfp.distributions.Dirichlet(
                    tf.fill(stats['P'].shape, self._concentration)).sample()
        
        if node.is_root():
            p = (1 - self._weight) * stats['P'] + self._weight * self._noise 
        else: 
            p = stats['P']
        
        return tf.argmax(puct(p, stats['Q'], stats['N'], node.get_n()))
        
    def reset(self) -> None:
        """Reset noise of this instance.
        
        Note:
            This method overrides `reset()` of `Selector`.
        """
        self._noise = None
  
      
class NodeVisitor():
    """Base node visitor.

    Node visitor conducts one simulation of mcts when traversing the tree.  
    
    Note:
        Descendants of `NodeVisitor` should override `check_terminal()`. They  
        also should call `__init__` of `NodeVisitor`.

    Attributes:
        _model (Model): Neural network model.
         _selector (Selector): Selector.
    """
    
    def __init__(self, model: Model, selector: Selector):
        """Initialize `NodeVisitor` instance.

        Args:
            model (Model): Neural network model.
            selector (Selector): Selector.
        """
        self._model = model
        self._selector = selector 
        
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
        """Update statistics of given node with given state value.

        Args:
            node (Node): The node in which this instance is located.
            v (float): The state value.

        Returns:
            float: State value of the current node in the perspective of its  
                parent.
        """
        return (-2 * self._is_multiagent + 1) \
                * self._selector.update_min_max_q(node.update_stats(v))
                
    def visit_leaf(self, node: Node) -> float:
        """Visit current leaf node.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the current leaf node.
        """
        return self._backup(node, self._expand_and_eval(node))
    
    def _expand_and_eval(self, node: Node) -> float:
        """Expand given node and evaluate state transition.

        Args:
            node (Node): Leaf node in which this instance is located.

        Returns:
            float: State value of the state transition.
        """
        if node.is_terminal():
            return 0
        
        if node.is_root():
            s = BaseAlphaZero.factory.create_state(*self._model.encode(node.get_o()))
            node.set_s(s)
        else: 
            r, s = self._model.process(node.get_intern_s(), node.get_a())
            node.set_r(r)
            node.set_s(s)
            
            if self.check_terminal(node):
                return 0
            
        P, v = self._model.estimate(s)
        
        for i, p in enumerate(P):
            node.add_child(
                    Node(s, BaseAlphaZero.factory.create_action(i), 
                         p, node.get_discount_factor()))
        
        return (-2 * self._is_multiagent + 1) * v 
    
    def check_terminal(self, node: Node) -> bool:
        """Check whether given node should be considered as terminal or not.

        Args:
            node (Node): The node in which this instance is located.

        Returns:
            bool: `True` if the node should be considered as terminal, `False`  
                otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  check_terminal().')
    
    def reset_selector(self) -> None:
        """Reset `selector` of this instance.
        """
        self._selector.reset()
    
    @classmethod
    def set_is_multiagent(cls, is_multiagent) -> None:
        cls._is_multiagent = is_multiagent
    
    def set_model(self, model: Model) -> None:
        self._model = model        


class BoundedVisitor(NodeVisitor):
    """Node visitior that stops expanding when it visits terminal node.
    """
    
    def __init__(self, model: Model, selector: Selector):
        """Initialize `BoundedVisitior` instance.

        Args:
            model (Model): Neural network model.
            selector (Selector): Selector.
        """
        super(BoundedVisitor, self).__init__(model, selector)
    
    def check_terminal(self, node: Node):
        """Check whether given node is terminal or not.
        
        Note: 
            This method overrides `check_terminal` of `NodeVisitor`.
        """
        return node.is_terminal()
