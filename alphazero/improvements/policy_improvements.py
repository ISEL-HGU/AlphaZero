"""Policy improvement interface and algorithms."""

import tensorflow as tf


class PolicyImprovement():
    """Base policy improvement.
    
    Note:
        - Descendants of `PolicyImprovement` should override `__call__()`.
    """
    
    def update_param(self, training_step: int) -> None:
        """Update parameters of this policy improvement algorithm.

        This method does not modify this instance.
        
        Args:
            training_step (int): Current traininig step.
        """
        pass
    
    def __call__(self, root) -> tf.Tensor:
        """Calculate improved policy using statistics of the given root node.

        Args:
            root (Node): The root node.
        
        Returns: 
            tf.Tensor: The improved policy.
        """
        raise NotImplementedError(f'class {self.__class__} did not override' 
                                   '__call__().')


class PVCD(PolicyImprovement):
    """Parameterized Visit Count Distribution policy improvement.

    Note:
        This class implements parameterized visit count distribution policy  
        improvement from "Mastering Chess and Shogi by Self-Play with a  
        General Reinforcement Learning Algorithm" `(Silver et al., '17)`_.
        
        .. _(Silver et al., '17): https://arxiv.org/abs/1712.01815
        
    Attributes:
        _temp (float): Temperature.
    """
    
    def __init__(self, temp: float = 1.0):
        """Initialize `PVCD` instance.

        Args:
            temp (float): Temperature.
        """
        self._temp = temp
    
    def update_param(self, training_step: int) -> None:
        """Update parameters of this policy improvement algorithm by halving  
        the temperature.

        Note:
            This method overrides `update_param()` of `PolicyImprovement`.
        """            
        self._temp /= 2
    
    def __call__(self, root) -> tf.Tensor: 
        """Improve policy by using children's visit counts of given root node.

        The policy is calculated by applying inverse proportion to children's  
        visit counts.  
        
        The formula used for calculating the policy is as follows:  
        .. math::
        - \\pi(a|s) = \\frac{N(s,a)^(1/\\tau)}{\\sum_{b} N(s,b)^(1/\\tau)},
            
        where (:math: N(s,\\dot)) and (:math: \\tau) denote visit count and  
        temperature respectively.

        Note: 
            This method overrides `__call__()` of `PolicyImprovement`.
        """
        pvc = root.obtain_stats_children()['N'] ** (1 / self._temp)
        
        return pvc / tf.reduce_sum(pvc)
