"""MDP reward."""

class Reward():
    """Base reward class.

    Note: 
        Descendants of `Reward` should call `__init__()` of `Reward` in the   
        constructor. Overriding `__str__()` can modify debugging message.
    
    Attributes:
        _val (float): Reward value.
    """
    
    def __init__(self, val: float):
        """Initialize `Reward` instance.

        Args:
            val (float): Reward value.
        """
        self._val = val
    
    def get_val(self) -> float:
        return self._val
