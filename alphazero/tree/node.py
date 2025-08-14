"""Tree node for MCTS."""

import tensorflow as tf

from alphazero.improvements.policy_improvements import PolicyImprovement
from alphazero.improvements.value_estimations.simulation_value_estimations \
        import SimulationValueEstimation
from alphazero.exceptions.node_exception import NodeException, NodeExceptionCode
from alphazero.mdp.action import Action 
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State 

class Node():
    """Tree node that conducts Monte-Carlo Tree Search (MCTS).  

    Each node is represented by internal state `s` and action `a`. Each node  
    stores set of statistics `{R(s,a), S(s,a), P(s,a), Q(s,a), N(s,a)}`, where
    - `R(s,a)` is intermediate reward.
    - `S(s,a)` is state transition.
    - `P(s,a)` is prior probability.
    - `Q(s,a)` is empirical state-action value.
    - `N(s,a)` is visit count. 
    
    The node is classified as following:
    - **Root node**: Node that is located at the top of the search tree.
    - **Expanded node**: Node that has children.
    - **Unexpanded node**: Node that does not have children.
    - **Terminal node**: Unexpanded node whose state transition is terminal  
      state.
    - **Undetermined node**: Unexpanded node that does not have state  
      transition yet.

    Attributes:
        _intern_s (State): The internal state.
        _a (Action): The action. 
        _r (Reward): The immediate reward. 
        _o (Observation): Observation of the state transition.
        _s (State): The state transition.
        _p (float): The prior probability. 
        _q (float): The empirical state-action value.
        _n (int): The visit count.
        _children (list of Node): Children nodes.
        _pi (PolicyImprovementStrategy): Policy improvement algorithm.
        _sve (SimulationValueEstimation): Simulation value  estimation  
            algorithm.
    """

    def __init__(self, intern_s: State | None, a: Action | None, 
                 p: float | None):
        """ Initialize `Node` instance.
        
        Args:
            intern_s (State): Internal state.
            a (Action): Action.
            p (float): Prior probability.
        """
        self._intern_s = intern_s
        self._a = a
        self._r = None
        self._o = None
        self._s = None
        self._p = p
        self._q = 0
        self._n = 0
        self._children = []
        self._pi = None
        self._sve = None
    
    @classmethod
    def root(cls, pi: PolicyImprovement, sve: SimulationValueEstimation) \
            -> 'Node':
        """Create root `Node` instance.
        
        Args:
            pi (PolicyImprovement): Policy improvement algorithm.
            tve (SimulationValueEstimation): Simulation demension value  
                estimation algorithm.
            
        Returns:
            Node: The root `Node` instance.
        """
        node = cls(None, None, None)
        node._pi = pi
        node._sve = sve
        
        return node
    
    def mcts(self, o: Observation, simulations: int, visitor) \
            -> tuple[tf.Tensor, float]:
        """Conduct Monte-Carlo Tree Search for the given number of simulations.

        Args:
            o (Observation): Observation of this instance.
            simulations (int): The number of simulations.
            visitor (NodeVisitor): Visitor that visits tree nodes.
        
        Returns:
            tuple: Improved policy and value estimate.
        
        Raises: 
            NodeException: If this instance is non-root.
        """
        if not self.is_root():
            raise NodeException(self.mcts.__qualname__, 
                                NodeExceptionCode.NONROOT)
        
        if not self.is_expanded():
            self._o = o
        
        for _ in range(simulations if self.is_expanded() else simulations + 1):
            self._accept(visitor)
        
        return (self._pi(self), self._sve(self))
        
    def _accept(self, visitor) -> float:
        """Accept a node visitor to this instance.

        The operation is to conduct one simulation of MCTS. The visitor  
        conducts pre visit internal operation when it reaches an expanded  
        node. The pre visit internal operation returns the index of child  
        node to visit. The visitor then moves to the child node and condcuts  
        the same process until it reaches the unexpanded node. When the  
        visitor reaches unexpanded node, it conducts vist leaf opearation.  
        After the operation, it moves back to the visited parent node and  
        conducts post visit internal operation. The visitor conducts the same  
        process until it reaches the root node.
           
        Args:
            visitor (NodeVisitor): The visitor.

        Returns:
            float: State-action value of this instance.
        """
        if not self.is_expanded():
            return visitor.visit_leaf(self)

        return visitor.post_visit_internal(
                self, 
                self._children[visitor.pre_visit_internal(self)]
                    ._accept(visitor))
    
    def make_root(self, pi: PolicyImprovement, 
                  sve: SimulationValueEstimation) -> None:
        """Make this instance into root node.

        See also:
            :meth:`alphazero.tree.node.Node.is_root`
        
        Args:
            pi (PolicyImprovement): Policy improvement algorithm.
            sve (SimulationValueEstimation): Simulation demension value  
                estimation algorithm.
        """
        self._pi = pi
        self._sve = sve
    
    def add_child(self, child: 'Node') -> None:
        """Add given child node to this instance.

        Args:
            child (Node): The child node.
        """
        self._children.append(child)

    def update_stats(self, v: float) -> float:
        """Update statistics of this instance with the given state value.
        
        Args:
            v (float): The state value of the state transition of this  
                instance.
        
        Returns:
            float: The state value if this instance is root, calculated  
                state-action value otherwise. 
        
        Raises:
            NodeException: If this intance is undetermined.
        """
        if not self.is_expanded() and not self.is_terminal():
            raise NodeException(self.update_stats.__qualname__, 
                                NodeExceptionCode.UNDETERMINED)
        
        if self.is_root():
            self._n += 1
            
            return v
        else:    
            g = self._r + self._r.get_discount_factor() * v
            
            self._n += 1
            self._q = self._q + (g - self._q) / self._n
            
            return g
    
    def obtain_stats_children(self) -> dict:
        """Obtain statistics of children.
        
        Returns:
            dict: The statistics.
        
        Raises:
            NodeException: If this instance is not expanded.
        """
        if not self.is_expanded():
            raise NodeException(self.obtain_stats_children.__qualname__, 
                                NodeExceptionCode.UNEXPANDED)
        
        P = []
        Q = []
        N = []
        
        for child in self._children:
            P.append(child._p)
            Q.append(child._q)
            N.append(child._n)
        
        return {'P': tf.constant(P), 'Q': tf.constant(Q), 'N': tf.constant(N)}
    
    def reset(self) -> None:
        """Reset statistics, children, and algorithms of this instance.
        """
        self._intern_s = None
        self._a = None
        self._r = None
        self._o = None
        self._s = None
        self._p = None
        self._q = 0
        self._n = 0
        self._children = []
        self._pi = None
        self._sve = None
        
    def is_root(self) -> bool:
        """Check whether this instance is root or not.

        Returns:
            bool: `True` if this instance is root, `False` otherwise.
        """
        return self._pi is not None and self._sve is not None
    
    def is_expanded(self) -> bool:
        """Check whether this instance is expanded or not.
        
        Returns:
            bool: `True` if this instance is expanded, `False` otherwise.
        """ 
        return (self.is_root() or isinstance(self._r, Reward)) \
                and isinstance(self._s, State) \
                and len(self._children) > 0
    
    def is_terminal(self) -> bool:
        """Check whether this instance is terminal or not.

        Returns:
            bool: `True` if this instance is terminal, `False` otherwise.
        """
        return self._r is not None \
                and (self._s is None or self._s.is_terminal()) \
                and len(self._children) == 0
    
    def get_child(self, i: int) -> 'Node':
        """Get the child node of given index.
        
        Args:
            i (int): The index.
    
        Returns:
            Node: The child node.
            
        Raises: 
            NodeException: If this instance is unexpanded.
            IndexError: If the index is bigger than or equal to the number of  
                children of this instance.
        """
        if not self.is_expanded():
            raise NodeException(self.get_child.__qualname__, 
                                NodeExceptionCode.UNEXPANDED)
        
        try:
            return self._children[i]
        except IndexError:
            raise IndexError(f'can only get 0th to {len(self._children) - 1}th' 
                             f' (not "{i}") child')
    
    def get_intern_s(self) -> State | None:
        return self._intern_s

    def get_a(self) -> Action | None:
        return self._a
    
    def get_o(self) -> Observation | None:
        return self._o
    
    def get_n(self) -> int:
        return self._n
    
    def set_r(self, r: Reward) -> None:
        self._r = r
    
    def set_s(self, s: State) -> None:
        self._s = s
