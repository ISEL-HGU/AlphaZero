"""Replay buffer."""

import collections

import tensorflow as tf


class ReplayBuffer():
    """Replay buffer that saves generated episode data.

    Attributes:
        _histories (collections.deque): Episode data.
    """    
    
    def __init__(self, capacity: int):
        """Initialize `ReplayBuffer` instance.

        Args:
            capacity (int): Capacity.
        """
        self._histories = collections.deque(maxlen=capacity)
    
    def add_history(self, history: list[dict[str, object]]) -> None:
        """Add given history to this instance.

        Args:
            history (list): The history.
        """
        self._histories.append(history)    
     
    def sample(self, n: int) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """Samples given number of random data from this instance.

        Args:
            n (int): The number of data to be sampled.
        
        Returns:
            tuple: Input data and output target. Input data is composed of  
                obseravtion and action representations. Output target is  
                composed of target policies, state values, and immediate  
                rewards.
        """
        observations = []
        actions = []
        policies = []
        state_values = []
        rewards = []
        
        for i in tf.random.uniform([n], maxval=len(self._histories), 
                                   dtype=tf.int32):
            j = tf.random.uniform([], maxval=len(self._histories[i]), 
                                  dtype=tf.int32) \
                         .numpy()
            observations.append(self._histories[i][j]['o'].get_repr())
            actions.append(self._histories[i][j]['a'].to_repr())
            policies.append(self._histories[i][j]['pi'])
            state_values.append(tf.constant(self._histories[i][j]['z']))
            rewards.append(tf.constant(self._histories[i][j]['u'].get_val()))
        
        return ([tf.stack(observations), tf.stack(actions)], 
                [tf.stack(policies), tf.stack(state_values), 
                 tf.stack(rewards)])
