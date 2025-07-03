"""MDP state and its observation."""

import tensorflow as tf

class Observation():
    """Base observation class.

    Note:
        Descendants of `Observation` should override `is_terminal()` and call  
        `__init__()` of `Observation` in the constructor.
    
    Attributes:
        _repr (tf.Tensor): Observation representation.
    """

    def __init__(self, repr: tf.Tensor):
        """Initialize `Observation` instance.

        Args:
            val (tf.Tensor): Observation representation.
        """
        self._repr = repr

    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal observation or not.

        This method will be called by `Environment` by the alphazero framework  
        and determine the episode termination.

        Returns:
            bool: `True` if this instance represents terminal observation,  
                `False` otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  is_terminal().')
    
    def get_repr(self) -> tf.Tensor:
        return self._repr
        
class State():
    """Base state class.  
    
    Note: 
        Descendants of `State` should override `is_terminal()`.
    
    Attributes:
        _repr (tf.Tensor): State representation.
    """
    
    def __init__(self, repr: tf.Tensor):
        """Initialize `State` instance.

        Args:
            val (tf.Tensor): State representation.
        """
        self._repr = repr
    
    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal state or not.

        Returns:
            bool: `True` if this instance represents terminal state, `False`  
                otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  is_terminal().')
    
    def get_repr(self) -> tf.Tensor:
        return self._repr
