"""MDP action."""

import tensorflow as tf

class Action():
    """Base action class.

    Note:
        Descendants of `Action` should override `to_arr()` if `Model` uses  
        dynamics network. Overriding `__str__()` can modify debugging message.
    
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
        """Convert this instance into an array representation.

        Note:
            This method is intended to be called in dynamics network only. If  
            simulator is used as dynamics of `Model`, this method should not  
            be called.  
        
        Returns:
            tf.Tensor: The array representation.
        """ 
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  to_arr().')
        
    def get_val(self) -> int:
        return self._val
