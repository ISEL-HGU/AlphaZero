"""MDP factory."""

import tensorflow as tf

from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State

class MDPFactory():
    """Abstract factory that creates `Observation`, `State`, `Action`, and  
    `Reward` instances.

    Note:
        Descendants of `MDPFactory` should override `create_observation`,  
        `create_state()`, `create_action()` and `create_reward()`.
    """

    def create_observation(self, repr: tf.Tensor) -> Observation:
        """Create `Observation` instance.

        Args:
            repr (tf.Tensor): Observation representation.
        
        Returns:
            Observation: The `Observation` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_observation().')

    def create_state(self, repr: tf.Tensor) -> State:
        """Create `State` instance.

        Args:
            repr (tf.Tensor): State representation.
            
        Returns:
            State: The `State` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_state().')
        
    def create_action(self, num: int) -> Action:
        """Create `Action` instance.
        
        Args:
            num (int): Action number.
        
        Returns:
            Action: The `Action` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_action().')
    
    def create_reward(self, *args) -> Reward:
        """Create `Reward` instance.

        Args:
            *args: Arguments for creating `Reward` instance.
            
        Returns:
            Reward: The `Reward` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  create_reward().')
