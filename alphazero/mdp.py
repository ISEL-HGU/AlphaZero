"""MDP definitions.
"""

import numpy as np

class Observation():
    """MDP observation interface.
   
    Class that extend this class should implement `is_terminal()` method. 
    """

    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal observation or not.

        Returns:
            bool: `true` if this instance represents terminal obervation,  
                `false` otherwise.
        """
        raise NotImplementedError(f'class {self.__class__.__name__} did not \
                                  implement is_terminal().')

class State():
    """MDP state interface.  
    
    Class that implements this interface should override `_check_terminal()`   
    and `_create_init_val()`.
    
    Attributes:
        _is_terminal (bool): Boolean value that indicates whether this   
                instance is terminal or not.
        _val (np.ndarray): State value.
    """
    
    def __init__(self, val: np.ndarray):
        """Initialize `State` instance.

        Args:
            val (np.ndarray): State value to be initialized.
        """
    
    def _check_terminal(self, val: np.ndarray) -> bool:
        """Check whether `val` is terminal or not.

        Args:
            val (np.ndarray): State value to be checked.

        Returns:
            bool: `True` if `val` is terminal, `False` otherwise. 
        """ 
    
    @classmethod
    def initial(cls, *args) -> 'State':
        """Create `State` instance that represents initial state. 
        
        Args:
            *args: Arguments for creating initial state value.

        Returns:
            State: The `State` instance.
        """
        
    @classmethod
    def _create_init_val(cls, *args) -> np.ndarray:
        """Create initial state value.

        Args:
            *args: Arguments for creating the state value.

        Returns:
            np.ndarray: The state value.
        """
    
    def get_is_terminal(self) -> bool:
        """Get boolean value that indicates whether this instance is terminal   
        or not.

        Returns:
            bool: `True` if this instance is terminal state, `False` otherwise.
        """
    
    def get_val(self) -> np.ndarray:
        """Get state value.

        Returns:
            np.ndarray: The state value.
        """

class Action():
    """MDP action interface.
    
    Attributes:
        _val (int): Action value.
    """
    
    def __init__(self, val: int):
        """Initialize `Action` instance.
        
        Args:
            val (int): Action value to be initialized.
        """    
             
class Reward():
    """MDP reward interface.
    
    Attributes:
        _val (float): Reward value.
        __add__ (function): Add magic method.
    """
    
    def __init__(self, val: float):
        """Initialize `Reward` instance.

        Args:
            val (float): Reward value to be initialized.
        """
    
    def __add__(self, other):
        """Add magic method.""" 

class MDPFactory():
    """Abstract factory that creates `State`, `Action`, and `Reward` instances.
    
    
    """