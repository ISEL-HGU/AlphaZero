"""MDP definitions.
"""
import tensorflow as tf

class Observation():
    """Base observation class.  
    
    Class that extends this class should override `is_terminal()` and  
    `to_args()`.
    
    Attributes:
        _val (tf.Tensor): Observation array.
    """

    def __init__(self, val: tf.Tensor):
        """Initialize `Observation` instance.

        Args:
            val (tf.Tensor): Observation array.
        """
        self._val = val

    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal observation or not.

        Returns:
            bool: `True` if this instance represents terminal obervation,  
                `False` otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  is_terminal().')
    
    def to_args(self) -> tuple:
        """Convert this instance into arguments for creating state.

        Returns:
            tuple: Arguments for creating state.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  to_args().')
    
    def get_val(self) -> tf.Tensor:
        return self._val
        
class State():
    """Base state class.  
    
    Class that extends this class should override `_check_terminal()`  
    and `_create_init_val()`.
    
    Attributes:
        _val (tf.Tensor): State array.
    """
    
    def __init__(self, val: tf.Tensor):
        """Initialize `State` instance.

        Args:
            val (tf.Tensor): State array.
        """
        self._val = val
    
    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal state or not.

        Returns:
            bool: `True` if this instance represents terminal state, `False`  
                otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  is_terminal().')
    
    def get_val(self) -> tf.Tensor:
        return self._val

class Action():
    """Base action class.
    
    Class that extends this class should override `to_arr()`.  
    Overriding `__str__()` helps debugging since training and inference logic  
    use the method for debugging purpose.
    
    Attributes:
        _val (int): Action number.
    """
    
    def __init__(self, val: int):
        """Initialize `Action` instance.
        
        Args:
            val (int): Action number.
        """    
        self._val = val
    
    def to_arr(self) -> tf.Tensor:
        """Convert this instance into array representation.
        
        Returns:
            tf.Tensor: The array representation.
        """ 
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  to_arr().')
        
    def get_val(self) -> int:
        return self._val
             
class Reward():
    """Base reward class.  
    
    Class that extends this class should override `__add__()`.  
    Overriding `__str__()` helps debugging since training and inference logic  
    use the method for debugging purpose.
    
    Attributes:
        _val (float): Reward value.
    """
    
    def __init__(self, val: float):
        """Initialize `Reward` instance.

        Args:
            val (float): Reward value.
        """
        self._val = val
    
    def __add__(self, other):
        return self._val + other
    
class MDPFactory():
    """Abstract factory that creates `State`, `Action`, and `Reward` instances.
    
    Class that extends this class should override `create_state()`,   
    `create_action()` and `create_reward()`.
    """

    def create_state(self, *args) -> State:
        """Create `State` instance.

        Args:
            *args: Arguments for creating `State` instance.
            
        Returns:
            State: `State` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_state().')
        
    def create_action(self, num: int) -> Action:
        """Create `Action` instance.
        
        Args:
            num (int): Action number.
        
        Returns:
            Action: `Action` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_action().')
    
    def create_reward(self, *args) -> Reward:
        """Create `Reward` instance.

        Args:
            *args: Arguments for creating `Reward` instance.
            
        Returns:
            Reward: `Reward` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_reward().')
