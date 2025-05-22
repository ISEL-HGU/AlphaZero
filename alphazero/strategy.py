"""
"""
import tensorflow as tf

from alphazero.node import Node

class PolicyImprovementStrategy():
    """Base policy improvement strategy class.
    
    Note:
        Class that extends this class should override `improve()`.
    """
    
    def improve(self, node: Node) -> tf.Tensor:
        """Calculate policy using statistics of the given node.

        Args:
            node (Node): Root node.
        
        Returns: 
            tf.Tensor: Improved policy.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  improve().')

class VisitCountStrategy(PolicyImprovementStrategy):
    """Class that improves policy with visit counts.
    
    Attributes:
        _temp (float): Temperature.
    """
    
    def __init__(self, temp: float):
        """Initialize `VisitCountStrategy` instance.

        Args:
            temp (float): Temperature for calculating policy.
        """
        self._temp = temp
    
    def improve(self, node: Node) -> tf.Tensor: 
        """Calculate policy by using children's visit counts of the given node.
        
        The policy is calculated by applying inverse proportion to children's  
        visit counts.  
        
        The formula used for calculating the policy is as follows:  
        .. math::
        - \pi(a|s) = \\frac{N(s,a)^(1/\\tau)}{\sum_{b} N(s,b)^(1/\\tau)},
            
        where (:math: N(s, \dot)) and (:math: \\tau) denote visit count and   
        temperature respectively.

        Note: 
            This method overrides `improve()` of `PolicyImprovementStrategy`.
        """
        visit_cnts = tf.constant(
                [child.get_n() for child in node.get_children()])
        
        return visit_cnts ** (1 / self._temp) \
                          / tf.reduce_sum(visit_cnts ** (1 / self._temp))
                
class ValuePropagationStrategy():
    """Base value propagation strategy class. 
    
    Note:
        Class that extends this class should override `prop()`.
    """
    
    def prop(self, history: list) -> None:
        """Calculate state values of the visited states and add them to the  
        given history.

        Args:
            history (list): Episode history.
        """
        raise NotImplementedError(f'class {self.__class__} did not implement \
                                  prop().')
        
class FinalOutcomeStrategy(ValuePropagationStrategy):
    """Class that propagates state value with the final outcome.
    """
    
    def prop(self, history: list) -> None:
        """Regard the final outcome as all the state values of the visited  
        states and add them to the given history. 

        Note: 
            This method overrides `prop()` of `ValuePropagationStrategy`.
        """
        for data in history:
            data['v'] = history[-1]['r']
