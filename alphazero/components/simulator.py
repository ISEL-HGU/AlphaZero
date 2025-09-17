"""Simulator interface."""

from alphazero import mdp
from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State


class Simulator():
    """Simulator that outputs reward and next state.

    Note:
        - Descendants of this class should be decorated with  
          `@keras.saving.register_keras_serializable()`.
        - Descendants of this class should override 
          `simulate_on_observation()`.
        - Descendants of this class that overrides `Simulator.__init__()`  
          with some parameters should override `Simulator.get_config()`.
        - Descendatns of this class that overrides `Simulator.__init__()`
          with some parameters that contain custom type should override  
          `Simulator.from_config()`.
    """
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     
    def simulate_on_state(self, s: State, a: Action) -> tuple[Reward, State]:
        """Simulate given action on given state.
        
        Args:
            s (State): The state.
            a (Action): The action.
                
        Returns:
            tuple: Immediate reward and next state.
        """
        r_features, o_features = self.simulate_on_observation(
                mdp.factory.create_observation(s.get_repr()), a)
        
        return (mdp.factory.create_reward(*r_features),
                mdp.factory.create_state(
                        mdp.factory.create_observation(*o_features)
                                   .get_repr()))
    
    def simulate_on_observation(self, o: Observation, a: Action) \
            -> tuple[tuple, tuple]:
        """Simulate given action on given observation.

        Args:
            o (Observation): The current observation.
            a (Action): The action.

        Returns:
            tuple: features of next reward and next observation. Features of  
                observation is `None` if the action is illegal on the current  
                observation.
        """
        raise NotImplementedError(f'class {self.__class__} did not override ' 
                                  'simulate_on_observation()')
    
    def get_config(self):
        return {}
