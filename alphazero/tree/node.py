"""Tree node for MCTS."""

import tensorflow as tf

from alphazero.algorithms.policy_improvements.policy_improvement \
        import PolicyImprovement
from alphazero.algorithms.value_estimation.tree.tree_value_estimation \
        import TreeValueEstimation
from alphazero.mdp.action import Action 
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State 
from alphazero.tree.visitor import NodeVisitor

class Node():
    """Tree node that conducts Monte-Carlo Tree Search (MCTS).  

    Each node is represented by internal state `s` and action `a`. Each node  
    stores set of statistics `{R(s,a), S(s,a), P(s,a), Q(s,a), N(s,a)}`, where  
    - `R(s,a)` is intermidiate reward.
    - `S(s,a)` is next state.
    - `P(s,a)` is prior probability.
    - `Q(s,a)` is empirical state-action value.
    - `N(s,a)` is visit count. 
    
    The node is classified as following:
    - Root node: Node that located at the top of the search tree.
    - Expanded node: Internal node that has children.
    - Unexpanded node: Leaf node that does not have any children.
    - Terminal node: Unexpanded node that has terminal state.
    
    Attributes:
        _intern_s (State): The internal state.
        _a (Action): The action. 
        _r (Reward): The immediate reward. 
        _o (Observation): Observation.
        _s (State): The next state.
        _p (float): The prior probability. 
        _q (float): The empirical state-action value.
        _n (int): The visit count.
        _children (list of Node): Children nodes.
        _pi (PolicyImprovementStrategy): Policy improvement algorithm.
        _tve (TreeValueEstimation): Tree demension value estimation algorithm.
    """
   
    def __init__(self, intern_s: 'State | None', a: 'Action | None', 
                 p: 'float | None'):
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
        self._children = None
        self._pi = None
        self._tve = None
    
    @classmethod
    def initial_root(cls, pi: PolicyImprovement, 
             tve: TreeValueEstimation) -> 'Node':
        """Create initial root `Node` instance.

        Args:
            pi (PolicyImprovement): Policy improvement algorithm.
            tve (TreeValueEstimation): Tree demension value estimation  
                algorithm.

        Returns:
            Node: The initial root `Node` instance.
        """
        node = cls(None, None, None)
        node._pi = pi
        node._tve = tve
        
        return node
    
    def mcts(self, o: Observation, simulations: int, visitor: NodeVisitor) \
            -> 'tuple[tf.Tensor, float] | None':
        """Conduct monte-carlo tree search for the given number of simulations. 

        Args:
            o (Observation): Observation of this instance.
            simulations (int): The number of simulations.
            visitor (NodeVisitor): Visitor that visits tree nodes.
        
        Returns:
            tuple|None: Tuple of improved policy and value estimate if root  
                node calls this method, `None` otherwise. 
        """
        if not self.is_root():
            return  

        self._o = o
        
        for _ in range(simulations):
            self._accept(visitor)
        
        return self._pi.improve(self), self._tve.estimate(self)
    
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
        if self._is_expanded():
            return visitor.post_visit_internal(
                    self, 
                    self._children[visitor.pre_visit_internal(self)]
                        ._accept(visitor))
        else: 
            return visitor.visit_leaf(self)
    
    def add_child(self, child: 'Node') -> None:
        """Add child to this instance.

        Args:
            child (Node): Child node.
        """
        if not self._children:
            self._children = []
        
        self._children.append(child)
    
    def update_stats(self, g: float) -> float:
        """Update statistics of this instance with the given cumulative reward.
        
        Args:
            g (float): The cumulative reward.
        
        Returns: 
            float: The cumulative reward. 
        """
        self._q = (self._n * self._q + g) / (self._n + 1)
        self._n += 1
        
        return g
    
    def reset(self) -> None:
        """Reset statistics and children of this instance.
        """
    
    def is_root(self) -> bool:
        """Check whether this instance is root or not.

        Returns:
            bool: `True` if this instance is root, `False` otherwise.
        """
        return True if self._pi and self._tve else False
    
    def _is_expanded(self) -> bool:
        """Check whether this instance is expanded or not.
        
        Returns:
            bool: `True` if this instance is expanded, `False` otherwise.
        """ 
        return bool(self._children)

    def get_intern_s(self) -> 'State | None':
        return self._intern_s

    def get_a(self) -> 'Action | None':
        return self._a

    def get_r(self) -> 'Reward | None':
        return self._r
    
    def get_o(self) -> 'Observation | None':
        return self._o
    
    def get_p(self) -> 'float | None':
        return self._p
    
    def get_q(self) -> float:
        return self._q

    def get_n(self) -> int:
        return self._n
    
    def get_children(self) -> 'list | None':
        return self._children
    
    def set_a(self, a: Action) -> None:
        self._a = a
    
    def set_r(self, r: Reward) -> None:
        self._r = r
    
    def set_s(self, s: State) -> None:
        self._s = s
