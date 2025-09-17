"""Simulation value estimation interface and algorithms."""

import tensorflow as tf


class SimulationValueEstimation():
    """Base simulation dimension value estimation.
    
    Note:
        - Descendants of `SimulationValueEstimation` should override  
          `__call__()`.
    """
    
    def __call__(self, root) -> float:
        """Estimate state value from given root node.

        Args:
            root (Node): The root node.

        Returns:
            float: The state value.
        """
        raise NotImplementedError(f'class {self.__class__} did not override' 
                                  '__call__().')


class SoftZ(SimulationValueEstimation):
    """Soft-Z simulation dimension value estimation algorithm.
    
    Note:
        This class implements Soft-Z value target from "Value Targets in  
        Off-Policy AlphaZero: A New Greedy Backup" `(Willemsen et al., '22)`_.
        
        .. _(Willemsen et al., '22): https://link.springer.com/article/10.1007/s00521-021-05928-5
    """
    
    def __call__(self, root) -> float:
        """Estimate empirical state value from given root node.

        Note: 
            This method overrides `__call__()` of `SimulationValueEstimation`.
        """
        stats = root.obtain_stats_children()
        
        return tf.reduce_sum(stats['N'] * stats['Q']) / (root.get_n() - 1)
