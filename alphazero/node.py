import tensorflow as tf
import numpy as np
from math import sqrt
from simulator import Simulator

class Node():
    """Node of mcts tree.

    Configuration of `Node` should be done \\
    by using `Node.configure()` before the any instantiation.
    
    Statistics of `Node` instance are set to `None` \\
    if `is_terminal` is `True`, all zeros otherwise.
    
    Args:
        s (np.ndarray): State of this instance.
        p_s (np.ndarray): Prior possiblities of selecting action 
            at this instance's state.
        is_terminal (bool): Flag that indicates whether this instance 
            represents terminal state or not.
    """
    
    #class variables
    _model: tf.keras.Model
    _simulator: Simulator
    _type: int 
    _gamma: float
    
    #instance variables
    _s: np.ndarray
    _p_s: np.ndarray
    _q_s: np.ndarray
    _w_s: np.ndarray
    _n_s: np.ndarray
    _r_s: np.ndarray
    _children: np.ndarray
    _is_terminal: bool
   
    def __init__(self, s: np.ndarray, p_s: np.ndarray, is_terminal: bool):
        self._s = s
        self._p_s = p_s
        
        if is_terminal:
            self._q_s = None
            self._w_s = None
            self._n_s = None
            self._r_s = None
        else:      
            self._q_s = np.zeros(p_s.shape, dtype=float)
            self._w_s = np.zeros(p_s.shape, dtype=float)
            self._n_s = np.zeros(p_s.shape, dtype=int)
            self._r_s = np.zeros(p_s.shape, dtype=float)
        
        self._children = np.full(p_s.shape, fill_value=None, dtype=Node)
        self._is_terminal = is_terminal
 
    
    @classmethod
    def configure(cls, model: tf.keras.Model, simulator: Simulator, 
                  type: int, gamma: float=1.0) -> None:
        """Configures policy-value network, simulator, type, 
        and discount factor of `Node`. 
        
        Args:
            model (tf.keras.Model): The neural network model.
            simulator (Simulator): The simulator. 
            type (int): The type of `Node`. 
                `0` indicates single-agented and `1` indicates double-agented.
            gamma (float, optional): The discount factor. 
                It should be in range of `[0, 1]`. Gamma of `0` indicates \\ 
                considering immediate reward only while `1` indicates considering \\ 
                immediate reward and all the future rewards equally. Setted to \\
                `1.0` by default.
        """
        
        cls._model = model
        cls._simulator = simulator
        cls._type = type
        cls._gamma = gamma
    
    
    def mcts(self, n_sim: int, tau: float=1.0) -> np.ndarray:
        """Conducts `n_sim` number of simulations of monte-carlo tree search \\
        to this instance and calculates policy with the accumulated statistics.

        The algorithm simulates with `_search()` `n_sim` times \\
        and calculates policy with `_calc_policy()`.
        
        Args: 
            n_sim (int): The number of simulations conducted.
            tau (float, optional): Temperature variable for calculating policy. 
                It should be in range of `(0, 1]`. The larger `tau` gets, \\
                it flattens distribution of the policy \\
                and leads more exploration. The smaller `tau` gets, \\ 
                it leads more exploitation. It is setted to `1.0` by default.
        
        Returns:
            np.ndarray: The policy of this instance.
        """
        
        for _ in range(n_sim):
            self._search()
        
        return self._calc_policy(tau)
        
        
    def _select(self, c_puct: float=4.0) -> int:
        """Selects action at the state of this instance accroding to \\
        puct algorithm.
        
        The equation of puct algorithm is \\
        `a = argmax_a(Q(s, a) + U(s, a))`, where:
        * `U(s, a) = c_puct * P(s, a) * 
        sqrt(sum_b(N(s, b))) / (1 + N(s, a))`
        
        Args:
            c_puct (float, optional): The puct constant. 
                Bigger c_puct value makes exploration to be more considered and \\ 
                smaller c_puct value makes exploitation to be more considered. \\
                Approximately value of 4 is optimal. 
                https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
        
        Returns:
            int: Integer that indicates the selected action.
        """

        return (self._q_s + c_puct * self._p_s 
                * sqrt(self._n_s.sum(axis=0)) / (self._n_s + 1)).argmax(axis=0)
       
    
    def _expand_and_evaluate(self, a: int) -> float:
        """Expands new leaf node to this instance and evaluates \\ 
        prior probabilities and state value of new state.

        The new state is obtained by taking aciton `a` \\
        on the state of this instance. Immediate reward is stored \\
        in r_sa of this instance. New node is expanded to `a`th child of \\
        this instance.

        Args:
            a (int): The selected action of this instance.
            
        Returns:
            float: The state value of expanded node in its perspective.
            
        Raises:
            ValueError: Raises when terminal node calls this method.
        """
        
        if self._is_terminal:
            raise ValueError('Terminal node invoked _expand_and_evaluate().')
        
        r, s_prime = Node._simulator.simulate(self._s, a)
        p, v = Node._model(s_prime[np.newaxis, :], False)
        is_terminal = Node._simulator.is_terminal(s_prime)
        
        if is_terminal:
            p = None
            v = 0.0
        
        self._children[a] = Node(s_prime, p, is_terminal)
        self._r_s[a] = r
         
        return v
        
        
    def _backup(self, v: float, i: int) -> None:
        """Updates statistics of this instance's `i`th edge \\
        with value `v`.
        
        Statistics of `i`th edge is updated as following: 
        * `W(s, a) = W(s, a) + v`
        * `N(s, a) = N(s, a) + 1`
        * `Q(s, a) = W(s, a) / N(s, a)`
            
        Args:
            v (float): Value of this instance's selected action.
            i (int): Index of this instance's edge to be updated.
        """
        
        self._w_s[i] += v
        self._n_s[i] += 1
        self._q_s[i] = self._w_s[i] / self._n_s[i] 
        
          
    def _search(self) -> float:
        """Searchs future states along the tree whose this instance is root. 
        
        The algorithm selects action with `_select()` until it reachs \\
        leaf node. Then it expands leaf node with new node and evaluates \\ 
        value of the new node by `_expand_and_evaluate()`. Finally it backups \\
        statistics of the node it visited with `_backup()` from the leaf node \\
        to the root node.
        
        The action value passed to `_backup()` is calculated as following: \\
        `r_sa + gamma * v` if `Node` is single-agented \\
        `r_sa + gamma * (-v)` if `Node` is double-agented, where 
        
        * `r_sa` is immediate reward obtained when action `a` is taken \\
        on state `s`. 
        * `gamma` is discount factor.
        * `v` is the action value of child node in its perspective.
        
        Returns:
            float: Value of this instance's selected action in its perspective.
        """
        
        if self._is_terminal:
            return 0.0
        
        a = self._select()
        v = None
        
        if self._children[a] is None:
            v = self._expand_and_evaluate(a)
        else:
            v = self._children[a]._search()
        
        if Node._type == 1:
            v = -v
        
        v = self._r_s[a] + Node._gamma * v
        
        self._backup(v, a)
            
        return v
        
        
    def _calc_policy(self, temp: float) -> np.ndarray:
        """Calculates policy of this instance with the given `temp`.
    
        The equation of the policy is \\
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
    def get_model(cls) -> tf.keras.Model:
        return cls._model
    
    
    @classmethod
    def get_simulator(cls) -> Simulator:
        return cls._simulator
    
    
    @classmethod
    def get_type(cls) -> int:
        return cls._type
         
         
    def get_s(self) -> np.ndarray:
        return self._s     
              
           
    def get_child(self, i: int) -> 'Node': 
        """Gets `i + 1`th child of this instance. 

        Args:
            i (int): Index of child to be obtained.
        
        Returns:
            Node: The `i + 1`th child.
        """     
        
        return self._children[i]


    def get_is_terminal(self) -> int:
        return self._is_terminal

