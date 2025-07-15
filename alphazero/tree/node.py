"""Tree node for MCTS."""

import tensorflow as tf

from alphazero.algorithms.policy_improvements.policy_improvement \
        import PolicyImprovement
from alphazero.algorithms.value_estimation.tree.tree_value_estimation \
        import TreeValueEstimation
from alphazero.exceptions.node_exception import NodeException
from alphazero.mdp.action import Action 
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State 
from alphazero.tree.visitor import NodeVisitor


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
    - Root node: Node that is located at the top of the search tree.
    - Expanded node: Node that is visited at least once.
    - Unexpanded node: Node that is not visited yet.
    - Terminal node: Leaf node whose state transition is a terminal state.  
        Terminal node belongs to expanded node. Terminal state can be both  
        legal and illegal.
    
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
        _tve (TreeValueEstimation): Tree demension value estimation algorithm.
        _discount_factor(float): Discount factor.
    """
   
    def __init__(self, intern_s: 'State | None', a: 'Action | None', 
                 p: 'float | None', discount_factor: float):
        """ Initialize `Node` instance.
        
        Args:
            intern_s (State): Internal state.
            a (Action): Action.
            p (float): Prior probability.
            discount_factor (float): Discount factor.
        """
        self._intern_s = intern_s
        self._a = a
        self._r = None
        self._o = None
        self._s = None
        self._p = p
        self._q = 0
        self._n = 0
        self._children = None
        self._pi = None
        self._tve = None
        self._discount_factor = discount_factor
    
    @classmethod
    def root(cls, pi: PolicyImprovement, tve: TreeValueEstimation, 
             discount_factor: float) -> 'Node':
        """Create root `Node` instance.
        
        Args:
            pi (PolicyImprovement): Policy improvement algorithm.
            tve (TreeValueEstimation): Tree demension value estimation  
                algorithm.
            discount_factor: Discount factor.
            
        Returns:
            Node: The root `Node` instance.
        """
        node = cls(None, None, None, discount_factor)
        node._pi = pi
        node._tve = tve
        
        return node
    
    def mcts(self, o: Observation, simulations: int, visitor: NodeVisitor) \
            -> 'tuple[tf.Tensor, float]':
        """Conduct monte-carlo tree search for the given number of simulations. 

        Args:
            o (Observation): Observation of this instance.
            simulations (int): The number of simulations.
            visitor (NodeVisitor): Visitor that visits tree nodes.
        
        Returns:
            tuple: Improved policy and value estimate.
        
        Raises: 
            NodeException: If this instance is non root.
        """
        if not self.is_root():
            raise NodeException(self.mcts.__qualname__, 'root') 

        self._o = o
        
        for _ in range(simulations if self._is_expanded() else simulations + 1):
            self._accept(visitor)
        
        return (self._pi.improve(self), self._tve.estimate(self))
    
    def _accept(self, visitor: NodeVisitor) -> float:
        """Accept visitor to this instance.

        The visitor selects child nodes until it visits leaf node. Then it  
        expands and evaluates the leaf node. Finally, it backups statistics of  
        all the visited nodes from leaf to root.

        Args:
            visitor (NodeVisitor): The visitor.

        Returns:
            float: State-action value of this instance.
        """
        if not self._is_expanded() or self.is_terminal():
            return visitor.visit_leaf(self)

        return visitor.post_visit_internal(
                self, 
                self._children[visitor.pre_visit_internal(self)]
                    ._accept(visitor))
    
    def make_root(self, pi: PolicyImprovement, tve: TreeValueEstimation) \
            -> None:
        """Make this instance into root node.

        See also:
            :meth:`alphazero.tree.node.Node.is_root`
        
        Args:
            pi (PolicyImprovement): Policy improvement algorithm.
            tve (TreeValueEstimation): Tree demension value estimation  
                algorithm.
        """
        self._pi = pi
        self._tve = tve
    
    def add_child(self, child: 'Node') -> None:
        """Add given child node to this instance.

        Args:
            child (Node): The child node.
        """
        if not self._children:
            self._children = []
        
        self._children.append(child)

    def update_stats(self, v: float) -> float:
        """Update statistics of this instance with the given state value.
        
        Args:
            v (float): The state value.
        
        Returns:
            float: The state value if this instance is root, cumulative  
                reward otherwise. 
        
        Raises: 
            NodeException: If this intance is not expanded.
        """
        if not self._is_expanded():
            raise NodeException(self.update_stats.__qualname__, 'expanded')
        
        if self.is_root():
            self._n += 1
            
            return v
        else: 
            g = self._r + self._discount_factor * v
            
            self._n += 1
            self._q = self._q + (g - self._q) / self._n
            
            return g
    
    def obtain_stats_children(self) -> 'dict[str, tf.Tensor]':
        """Obtain statistics of children.
        
        Returns:
            dict: The statistics.
        
        Raises:
            NodeException: If this instance is not expanded or terminal.
        """
        if not self._is_expanded():
            raise NodeException(self.obtain_stats_children.__qualname__, 
                                'expanded')
        
        if self.is_terminal():
            raise NodeException(self.obtain_stats_children.__qualname__, 
                                'non terminal')
        
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
        self._children = None
        self._pi = None
        self._tve = None
        
    def is_root(self) -> bool:
        """Check whether this instance is root or not.

        Returns:
            bool: `True` if this instance is root, `False` otherwise.
        """
        return self._pi is not None and self._tve is not None
    
    def is_terminal(self) -> bool:
        """Check whether this instance is terminal or not.

        Returns:
            bool: `True` if this instance is terminal, `False` otherwise.
        """
        return self._is_expanded() and self._s.is_terminal()
    
    def _is_expanded(self) -> bool:
        """Check whether this instance is expanded or not.
        
        Returns:
            bool: `True` if this instance is expanded, `False` otherwise.
        """ 
        return self._s is not None
    
    def get_child(self, i: int) -> 'Node':
        """Get the child node of given index.
        
        Args:
            i (int): The index.
    
        Returns:
            Node: The child node.
            
        Raises:
            NodeException: If this instance is not expanded or terminal.
            IndexError: If the index is out of range.
        """
        if not self._is_expanded():
            raise NodeException(self.get_child.__qualname__, 'expanded')
        
        if self.is_terminal():
            raise NodeException(self.get_child.__qualname__, 'non terminal')
        
        if i < 0 or i >= len(self._children):
            raise IndexError(f'can only get 0th to {len(self._children) - 1}th \
                             child (not "{i}").')
            
        return self._children[i]

    def get_intern_s(self) -> 'State | None':
        return self._intern_s

    def get_a(self) -> 'Action | None':
        return self._a
    
    def get_o(self) -> 'Observation | None':
        return self._o
    
    def get_n(self) -> int:
        return self._n
    
    def get_discount_factor(self) -> float:
        return self._discount_factor
    
    def set_r(self, r: Reward) -> None:
        self._r = r
    
    def set_s(self, s: State) -> None:
        self._s = s
