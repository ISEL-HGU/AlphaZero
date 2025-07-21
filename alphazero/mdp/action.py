"""MDP action."""

import tensorflow as tf

class Action():
    """Base action class.

    Note:
        Descendants of `Action` should override `to_repr()` if `Model` uses  
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
    
    def to_repr(self) -> tf.Tensor:
        """Convert this instance into the action representation.

        Note:
            This method is intended to be called only when the `Model` uses  
            dynamics network. If simulator is used for dynamics of `Model`,  
            this method should not be called.  
        
        Returns:
            tf.Tensor: The action representation.
        """ 
        raise NotImplementedError(f'class {self.__class__} did not override' 
                                   'to_repr().')
        
    def get_num(self) -> int:
        return self._num
