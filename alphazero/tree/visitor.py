"""Node visitors."""

import tensorflow as tf
import tensorflow_probability as tfp

from alphazero import mdp
from alphazero.exceptions.node_exception \
        import NodeException, NodeExceptionCode
from alphazero.nn.model import Model
from alphazero.tree.node import Node
from alphazero.tree.utils import puct

          
class NodeVisitor():
    """Base node visitor.

    Node visitor that conducts one simulation of mcts when traversing the tree.  
    
    Note:
        - Descendants of `NodeVisitor` should call `__init__()` in their
          constructor.
        - Descendants of `NodeVisitor` should override `_select()`.
        - Descendatns of `NodeVisitor` can override `reset()` for custom
          instance reset.

    Attributes:
        _model (Model): Neural networks model.
        _min_q (float): Minimum state-action value of the search tree.
        _max_q (float): Maximum state-action value of the search tree.
        _is_multiagent (bool): Flag of multi-agent perspective. 
    """
    
    def __init__(self, model: Model, is_multiagent: bool):
        """Initialize `NodeVisitor` instance.

        Args:
            model (Model): Neural networks model.
            is_multiagent (bool): Flag that indicates whether this instance   
                visits nodes in multi-agent perspective or not.
        """
        self._model = model
        self._min_q = float('inf')
        self._max_q = float('-inf')
        self._is_multiagent = is_multiagent
        
    def pre_visit_internal(self, node: Node) -> int:
        """Conduct the operation on given internal node whose descendants are  
        not visited yet.
        
        The operation is to select an action number of the child node to visit.
        
        Args: 
            node (Node): The internal node in which this instance is located.
            
        Returns:
            int: The action number.
        """
        return self._select(node)

    def _select(self, node: Node) -> int:
        """Select the action number of given internal node's child.

        Args:
            node (Node): The internal node.

        Returns:
            int: The action number.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                   '_select().')
        
    def post_visit_internal(self, node: Node, v: float) -> float:
        """Conduct the operation on given expanded node whose selected  
        descendants are visited.
        
        The operation is to back up the node with given state value.

        Args:
            node (Node): The expanded node in which this instance is located.
            v (float): The state value of the node's state transition.

        Returns:
            float: State-action value of the node.
        
        Raises:
            NodeException: If the node is unexpanded.
        """
        if not node.is_expanded():
            raise NodeException(self.post_visit_internal.__qualname__, 
                                NodeExceptionCode.UNEXPANDED)
        
        return self._backup(node, v)
    
    def _backup(self, node: Node, v: float) -> float: 
        """Update statistics of given node with given state value.

        The minimum and maximum state-action value are updated. 

        Args:
            node (Node): The node.
            v (float): The state value.

        Returns:
            float: State-action value of the node in the perspective of its  
                parent.
                
        Raises:
            NodeException: If the node is undetermined.
        """
        q = node.update_stats(v)
        
        if q < self._min_q:
            self._min_q = q
        
        if q > self._max_q:
            self._max_q = q
        
        return (-2 * self._is_multiagent + 1) * q
                
    def visit_leaf(self, node: Node) -> float:
        """Visit given unexpanded node.

        The operation is to expand the node and evaluate the state value of  
        the node's state transition.

        Args:
            node (Node): The unexpanded node in which this instance is located.

        Returns:
            float: State-action value of the unexpanded node in the  
                perspective of its parent.
        
        Raises:
            NodeException: If the node is expanded or terminal root or  
                undetermined root that turns out to be a terminal.
        """
        return self._backup(node, self._expand_and_eval(node))
    
    def _expand_and_eval(self, node: Node) -> float:
        """Expand given unexpanded node and evaluate its state transition.

        Args:
            node (Node): The unexpanded node in which this instance is located.

        Returns:
            float: State value of the state transition.
        
        Raises:
            NodeException: If the node is expanded.
        """
        if node.is_expanded():
            raise NodeException(self.visit_leaf.__qualname__, 
                                NodeExceptionCode.EXPANDED)
        
        if node.is_terminal():
            return 0
        
        if node.is_root():
            s = self._model.encode(node.get_o())
            node.set_s(s)
        else: 
            r, s = self._model.process(node.get_intern_s(), node.get_a())
            node.set_r(r)
            node.set_s(s)
        
        if s is None or s.is_terminal():
            return 0
        
        P, v = self._model.estimate(s)
        
        for i, p in enumerate(P):
            node.add_child(Node(s, mdp.factory.create_action(i), p))
        
        return (-2 * self._is_multiagent + 1) * v 
    
    def reset(self) -> None:
        """Reset states of this instance.
        """
        pass
    
    def set_model(self, model: Model) -> None:
        self._model = model


class PUCTVisitor(NodeVisitor):
    """Node visitor that selects a child with PUCT algorithm.

    modifies prior probabilities with Dirichlet noise and  

    Attributes:
        _noise (tf.Tensor): Dirihclet noise
        _concentration (float): Concenctration parameter.
        _noise_weight (float): Noise weight.
    """
    
    def __init__(self, model: Model, concentration: float, 
                 noise_weight: float, is_multiagent: bool):
        """Initialize `PUCTVisitor` instance.

        Args:
            model (Model): Neural networks model.
            concentration (float): Concentration parameter of dirichlet  
                distribution.
            noise_weight (float): Dirichlet noise weight.
            is_multiagent(bool): Flag that indicates whether this instance  
                visits nodes in multi-agent perspective or not.
        """
        super(PUCTVisitor, self).__init__(model, is_multiagent)
        
        self._noise = None
        self._concentration = concentration
        self._noise_weight = noise_weight

    def _select(self, node: Node) -> int:
        """Select the action number of given internal node's child by using  
        PUCT algorithm.

        Note: 
            This method overrides `NodeVisitor._select()`.
        
        See Also:
            :py:meth:`~utils.puct`
        """
        stats = node.obtain_stats_children()
        
        if self._noise is None: 
            self._noise = tfp.distributions.Dirichlet(
                    tf.fill(stats['P'].shape, self._concentration)).sample()
        
        if node.is_root():
            p = (1 - self._noise_weight) * stats['P'] \
                    + self._noise_weight * self._noise 
        else: 
            p = stats['P']
        
        return tf.argmax(puct(p, stats['Q'], stats['N'], node.get_n()))
        
    def reset(self) -> None:
        """Reset noise of this instance.
        
        Note:
            This method overrides `NodeVisitor.reset()`.
        """
        self._noise = None
