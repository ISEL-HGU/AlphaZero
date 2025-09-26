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
            tuple: Input and target data. Input data is composed of  
                obseravtion and action representations. Target data is  
                composed of target policies, state values, and immediate  
                rewards.
        """
        observations = []
        actions = []
        policies = []
        state_values = []
        rewards = []
          
        for history in [self._histories[i] 
                            for i in tf.random.uniform(
                                [n], maxval=len(self._histories),
                                dtype=tf.int32)]:
            i = tf.random.uniform([], maxval=len(history), dtype=tf.int32) \
                         .numpy()
            
            observations.append(history[i]['o'])
            actions.append(history[i]['a'])
            policies.append(history[i]['pi'])
            state_values.append(tf.constant(history[i]['z']))
            rewards.append(tf.constant(history[i]['u']))
        
        return ([tf.stack(observations), tf.stack(actions)],
                [tf.stack(policies), tf.stack(state_values), tf.stack(rewards)])

    def __len__(self):
        return len(self._histories)
