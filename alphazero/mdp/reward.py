"""MDP reward."""

class Reward():
    """Base reward class.

    Note: 
        Descendants of `Reward` should call `__init__()` of `Reward` in the   
        constructor. Overriding `__str__()` can modify debugging message.
    
    Attributes:
        _val (float): Reward value.
        _discount_factor (float): Discount factor.
    """
    
    def __init__(self, val: float, discount_factor):
        """Initialize `Reward` instance.

        Args:
            val (float): Reward value.
            discount_factor (float): Discount factor.
        """
        self._val = val
        self._discount_factor = discount_factor
    
    def get_val(self) -> float:
        return self._val
    
    def get_discount_factor(self) -> float:
        return self._discount_factor
    
    def __add__(self, other):
        return self._val + other

    def __str__(self):
        return str(self._val)
