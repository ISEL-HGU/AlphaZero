"""MDP action."""

import tensorflow as tf

class Action():
    """Base action class.

    Note:
        Descendants of `Action` should override `to_arr()` if `Model` uses  
        dynamics network, and call `__init__()` of `Action` in  the  
        constructor. Overriding `__str__()` can modify debugging message.
    
    Attributes:
        _num (int): Action number.
    """
    
    def __init__(self, num: int):
        """Initialize `Action` instance.
        
        Args:
            num (int): Action number.
        """    
        self._num = num
    
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
        
    def get_num(self) -> int:
        return self._num
