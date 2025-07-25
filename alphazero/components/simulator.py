"""Simulator interface."""

from alphazero.mdp.action import Action
from alphazero.mdp.factory import MDPFactory
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import State

class Simulator():
    """Simulator that outputs reward and next state.

    Note:
        Descendants of `Simulator` should override `simulate()`.
    
    Attributes:
        _factory (MDPFactory): MDP factory.
    """
    
    def __init__(self, factory: MDPFactory):
        self._factory = factory
        
    def simulate(self, s: State, a: Action) -> tuple[Reward, State]:
        """Simulate given action on given state.
        
        Args:
            s (State): The state.
            a (Action): The action.
                
        Returns:
            tuple: Immediate reward and next state.
        """
        raise NotImplementedError(f'class {self.__class__} did not override ' 
                                   'simulate()')
        