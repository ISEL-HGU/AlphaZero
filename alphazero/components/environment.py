"""Environment interface."""

from alphazero import mdp
from alphazero.components.simulator import Simulator
from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation


class Environment():
    """Environment that outputs reward and observation.

    Note:
        - Descendants of this class should override `_induce()`. 
        - Descendants of this class should call `Environment.__init__()` in  
          their `__init__()`.
    
    Attributes:
        _simulator (Simulator): Simulator.
        _r (Reward): Current reward.
        _o (Observation): Current observation.
        _t (int): Current time step.
        _t_limit (int): Maximum time step.
    """

    def __init__(self, simulator: Simulator, t_limit: int):
        """Initialize `Environment` instance.
        
        Args:
            simulator (Simulator): Simulator.
            t_limit (int): Maximum time step. `-1` represents infinite time  
                step.
        """
        self._simulator = simulator
        self._r = None
        self._o = mdp.factory.create_observation(start=True)
        self._t = 0
        self._t_limit = t_limit
        
    def apply(self, a: Action) -> None: 
        """Apply given action to the observation of this instance.

        Reward and observation of this instance are updated.

        Args:
            a (Action): The action.
        
        Raises:
            Exception: If this instance is already terminated and this method  
                is called.
        """
        if self.is_terminated():
            raise Exception('This instance is terminated. Call '
                            f'{self.reset.__qualname__} before calling '
                            f'{self.apply.__qualname__}.')
        
        r_features, o_features = self._simulator.simulate_on_observation(
                self._o, a)
        
        self._r = mdp.factory.create_reward(*r_features)
        self._o = None if o_features is None \
                       else mdp.factory.create_observation(*o_features)
    
    def reset(self) -> None: 
        """Reset this instance.
        """
        self._r = None
        self._o = mdp.factory.create_observation(start=True)
        self._t = 0
        
    def is_terminated(self) -> bool:
        """Checks whether this instance is terminated or not.

        Returns:
            bool: `True` if this instance is terminated, `False` otherwise.
        """
        return self._t == self._t_limit or self._o is None \
                or self._o.is_terminal()
    
    def get_r(self) -> Reward | None:
        return self._r
    
    def get_o(self) -> Observation | None:
        return self._o
