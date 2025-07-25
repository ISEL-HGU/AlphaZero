"""Environment interface."""

from alphazero.mdp.action import Action
from alphazero.mdp.factory import MDPFactory
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation

class Environment():
    """Environment that outputs reward and observation.

    Note:
        Descendants of `Environment` should override `induce()`. They should  
        also call `__init__()` of `Environment` in the constructor.
    
    Attributes:
        _r (Reward): Current reward.
        _o (Observation): Current observation.
        _t (int): Current time step.
        _factory (MDPFactory): MDP factory.
        _t_limit (int): Maximum time step.
    """

    def __init__(self, factory: MDPFactory, t_limit: int):
        """Initialize `Environment` instance.
        
        Args:
            factory (MDPFactory): MDP factory.
            t_limit (int): Maximum time step. `-1` represents infinite time  
                step.
        """
        self._r = None
        self._o = factory.create_observation(start=True)
        self._t = 0
        self._factory = factory
        self._t_limit = t_limit
        
    def apply(self, a: Action) -> None: 
        """Apply given action to the observation of this instance.

        Reward and observation of this instance are updated.

        Args:
            a (Action): The action.
        """
        r_args, o_args = self.induce(self._o, a)
        self._r = self._factory.create_reward(*r_args)
        self._o = self._factory.create_observation(*o_args)
        self._t += 1
        
    def induce(self, o: Observation, a: Action) -> tuple[tuple, tuple]:
        """Induce reward and next observation by using given current  
        observation and action.

        Args:
            o (Observation): The current observation.
            a (Action): The action.

        Returns:
            tuple: Arguments of reward and next observation.
        """
        raise NotImplementedError(f'class {self.__class__} did not override'
                                   'induce().')
    
    def reset(self) -> None: 
        """Reset this instance.
        """
        self._r = None
        self._o = self._factory.create_observation(start=True)
        self._t = 0
        
    def is_terminated(self) -> bool:
        """Checks whether this instance is terminated or not.

        Returns:
            bool: `True` if this instance is terminated, `False` otherwise.
        """
        return self._t == self._t_limit or self._o.is_terminal()
    
    def get_r(self) -> Reward | None:
        return self._r
    
    def get_o(self) -> Observation:
        return self._o
