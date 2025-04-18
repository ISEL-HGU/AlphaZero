import tensorflow as tf
import numpy as np
from math import sqrt
from mdp import Action, Observation
from alphazero.forwardables.simulator import Simulator

class Node():
    """Tree node that conducts Monte-Carlo Tree Search (MCTS).
    
    Attributes:
        _a (Action or int): Action of this instance. 
        _r (Reward): Immediate reward of this instance's action. 
        _s (State): State of this instance.
        _p (float): Prior probability of this instance's action. 
        _q (float): Average action value of this instance's action.
        _n (int): Visit count of this instance's action.
        _children (list of Node): Children nodes.
        _parent (Node): Parent node.
        _config (dict): Configurations.
    """
   
    def __init__(self, a: int, p: float, parent: "Node", config: dict):
        """ Initialize `Node` instance.
        
        Args:
            a_num (int): State of this instance.
            p (float): Prior probability of selecting this instance's action.
            parent (Node): Parent node.
            config (dict): Configurations.
        """
        self._a = a
        self._p = p
        self._q = 0
        self._n = 0
        self._children = []
        self._parent = parent
        self._config = config
     
    def mcts(self, o: Observation, simulations: int) -> np.ndarray:
        """Conducts `simulations` number of simulations of monte-carlo tree   
        search to this instance. 

        Args:
            o (Observation): Observation that this instance represents.
            simulations (int): The number of simulations of tree search   
                conducted.
        
        Returns:
            np.ndarray: The policy of this instance.
        """
        self._r = 0
        self._s = self._config['repr_strat'].encode(o)
    
        probs, v = self._config['pred_net'](self._s.to_arr())
        self._expand(probs)
        
                
        for _ in range(simulations):
            self._search()
        
        return self._calc_policy(tau)
        
        
    def _select(self) -> Action:
        """Select action according to the selection strategy of this instance.
        
        Returns:
            Action: Selected action.
        """
        return self._config['select_strat'].selectAction(self)
    
    def _expand(self, probs: np.ndarray) -> None:
        """Add all the child nodes to this instance.

        Args:
            probs (np.ndarray): Prior probabilities of the child nodes.
        """
        for i, p in enumerate(probs):
            self._children.append(Node(i, p, self, self._config))
    
    def _eval(self, v: float) -> float:
        """Calculate action value of this instance's action.
        
        Args: 
            v (float): State value of this instance's state.
        
        Returns:
            float: Action value of this instance's action.
        """ 
        return self._r + self._config['gamma'] * v
    
    def _expand_and_eval(self) -> float:
        """Expand new leaf nodes to this instance and evaluate action value  
        of this instance's action.

        State and immediate reward is stored in this instance before the  
        expansion and evaluation.
            
        Returns:
            float: Action value of this instance's action.            
        """
        r, s = self._config['conductor'].nextState(self._parent._s, self._a)
        
        self._r = r
        self._s = s
        
        probs, v = self._config['predictor'](s)
        self._expand(probs)
               
        return self._eval(v)
        
    def _backup(self, g: float) -> None:
        """Update statistics of this instance.
        
        Args:
            v (float): Value. 
        """
        self._q = (self._n * self._q + g) / (self._n + 1)
        self._n += 1
        
    def _is_expanded(self) -> bool:
        """ Check whether this instance is expanded or not.
        
        Returns:
            bool: `true` if this instance is expanded, `false` otherwise.
        """ 
        return True if len(self._children) else False
        
    def _search(self) -> float:
        """Conduct 1 simulation of MCTS.
        
        The function is defined as expanding new leaf nodes and update  
        statistics of all the nodes it visited.
        
        The action value passed to `_backup()` is calculated as following:  
        `r_sa + gamma * v` if `Node` is single-agented  
        `r_sa + gamma * (-v)` if `Node` is double-agented, where 
        
        * `r_sa` is immediate reward obtained when action `a` is taken 
        on state `s`. 
        * `gamma` is discount factor.
        * `v` is the action value of child node in its perspective.
        
        Returns:
            float: Value of this instance's selected action in its perspective.
        """
        if self._is_terminal:
            return 0.0
        
        if self._is_expanded():
            v = self._children[self._select().get_val()]._search()
        else: 
            v = self._expand_and_eval()
            
        self._backup(v)
            
        return v
         
    def _calc_policy(self, temp: float) -> np.ndarray:
        """Calculates policy of this instance with the given `temp`.
    
        The equation of the policy is 
        `pi = pow(N(s, .), 1 / tau) / sum_b(pow(N(s, b), 1 / tau))`, where:
        * `tau` indicates the temperature variable. 

        Args:
            temp (float): The temperature variable.  

        Returns:
            np.ndarray: The policy of this instance.
        """   

        n_pow = self._n_s ** (1 / temp) 

        return n_pow / n_pow.sum(axis=0)
    
    @classmethod
    def get_type(cls) -> int:
        return cls._type

    def get_is_terminal(self) -> bool:
        return self._is_terminal
