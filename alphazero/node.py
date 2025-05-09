"""
"""
import tensorflow as tf

from alphazero.mdp import Action, Observation, Reward, State 
from alphazero.strategy import PolicyImprovementStrategy
from alphazero.visitor import NodeVisitor

class Node():
    """Tree node that conducts Monte-Carlo Tree Search (MCTS).  
    
    Each node is represented by internal state and action pair.
    
    Attributes:
        _o (Observation): Observation of this instance.
        _intern_s (az.State): Internal state.
        _a (Action or int): Action of this instance. 
        _r (Reward): Immediate reward of this instance's action. 
        _s (State): State of this instance.
        _p (float): Prior probability of this instance's action. 
        _q (float): Average action value of this instance's action.
        _n (int): Visit count of this instance's action.
        _children (list of Node): Children nodes.
        _strat (PolicyImprovementStrategy): Policy improvement strategy.
    """
   
    def __init__(self, intern_s: State, a: int, p: float):
        """ Initialize `Node` instance.
        
        Args:
            intern_s: Internal state of this instance.
            a_num (int): Action number of this instance.
            p (float): Prior probability of selecting this instance's child.
        """
        self._o = None
        self._intern_s = intern_s
        self._a = a
        self._r = 0
        self._s = None
        self._p = p
        self._q = 0
        self._n = 0
        self._children = []
        self._strat = None
    
    @classmethod
    def root(cls, o: Observation, strat: PolicyImprovementStrategy) -> 'Node':
        """Create root `Node` instance.

        Args:
            o (Observation): Observation of root node.
            strat (dict): Policy improvement strategy.

        Returns:
            Node: Root `Node` instance.
        """
        node = cls(None, None, None)
        node._o = o
        node._strat = strat
        
        return node
    
    def mcts(self, simulations: int, visitor: NodeVisitor) -> tf.Tensor:
        """Conduct monte-carlo tree search for the given number of simulations. 

        Args:
            simulations (int): The number of simulations.
            visitor (NodeVisitor): Visitor that visits tree nodes.
        
        Returns:
            tf.Tensor: The policy of this instance.
        """
        for _ in range(simulations):
            self._accept(visitor)
        
        return self._strat.improve(self)
    
    def _accept(self, visitor: NodeVisitor) -> float:
        """Accept visitor to this instance.  
        
        The visitor selects child nodes until it visits leaf node. Then it  
        expands and evaluates the leaf node. Finally it backups statistics of  
        all the visited nodes from leaf to root.

        Args:
            visitor (NodeVisitor): Visitor.

        Returns:
            float: Action value of this instance.
        """
        if self._is_expanded():
            return visitor._post_visit_internal(self, 
                    self._children[visitor._pre_visit_internal(self)]
                        .accept(visitor))
        else: 
            return visitor._visit_leaf(self)
        
    def update(self, g: float) -> float:
        """Update statistics of this instance.
        
        Args:
            v (float): Emperical action value.
        
        Returns: 
            float: Updated action value. 
        """
        self._q = (self._n * self._q + g) / (self._n + 1)
        self._n += 1
        
        return g
    
    def add(self, child: 'Node') -> None:
        """Add child to this instance.

        Args:
            child (Node): Child node.
        """
        self._children.append(child)
    
    def is_root(self) -> bool:
        """Check whether this instance is root or not.

        Returns:
            bool: `True` if this instance is root, `False` otherwise.
        """
        return True if self._o else False
    
    def _is_expanded(self) -> bool:
        """Check whether this instance is expanded or not.
        
        Returns:
            bool: `True` if this instance is expanded, `False` otherwise.
        """ 
        return True if len(self._children) else False

    def get_o(self) -> Observation:
        return self._o

    def get_intern_s(self) -> State:
        return self._intern_s

    def get_a(self) -> Action or int:
        return self._a

    def get_r(self) -> Reward:
        return self._r
    
    def set_a(self, a: Action) -> None:
        self._a = a
    
    def set_r(self, r: Reward) -> None:
        self._r = r
    
    def set_s(self, s: State) -> None:
        self._s = s
