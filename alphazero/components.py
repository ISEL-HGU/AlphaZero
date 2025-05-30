"""Core components of AlphaZero.
"""

import numpy as np

from alphazero.mdp import Action, Observation, Reward
from alphazero.node import Node
from alphazero.replaymemory import ReplayMemory
from alphazero.strategy import ValuePropagationStrategy
from alphazero.visitor import NodeVisitor

class Environment():
    """Environment that outputs reward and next observation.  

    Note:
        Class that extends this class should override `obtain_start_args()`  
        and `observe`. 
    
    Attributes:
        _r (Reward): Reward.
        _o (Observation): Observation.
    """

    def __init__(self):
        """Initialize `Environment` instance.
        """
        self._set(*self.obtain_start_args())
    
    def obtain_start_args(self) -> tuple:
        """Obtain initial reward and observation arguments.
        
        Returns: 
            tuple: Tuple of initial reward and observation arguments.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  `obtain_start_args`.')
        
    def _set(self, r_args: tuple, o_args: tuple) -> None:
        """Set reward and observation of this instance.  
        
        `Reward` and `Observation` instance are created by using the given  
        arguments.
        
        Args:
            r_args (tuple): Reward arguments.
            o_args (tuple): Observation arguments.
        """
        self._r = Agent.factory.create_reward(*r_args)
        self._o = Agent.factory.create_observation(*o_args)
    
    def apply(self, a: Action) -> None: 
        """Apply the given action to this instance.

        Reward and observation of this instance are updated.

        Args:
            a (Action): The action.
        """
        self._set(*self.observe(a))
        
    def observe(self, a: Action) -> tuple:
        """Observe reward and next observation by using the given action.

        Args:
            a (Action): The action.

        Returns:
            tuple: Tuple of reward and observation arguments.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  `observe`.')
    
    def reset(self) -> None: 
        """Reset this instance.
        """
        self._set(*self.obtain_start_args())
    
    def is_terminated(self) -> bool:
        """Checks whether this instance is terminated or not.

        Returns:
            bool: `true` if this instance is terminated, `false` otherwise.
        """
        return self._o.is_terminal()
    
    def get_r(self) -> Reward:
        return self._r
    
    def get_o(self) -> Observation:
        return self._o
        
class Agent():
    """Agent that acts on an environment.  
    
    Note:
        Class that extends this class should override  
        `_create_train_p_imprv_strat()` and `create_infer_p_imprv_strat()`.
    
    Attributes:
        _history (list): History of the episode.
        _visitor (NodeVisitor): Node visitor.  
        _strat (ValuePropagationStrategy): Value propagation strategy.
        _simulations (int): The number of simulations.
        _is_training (bool): Value that indicates whether this instance is  
            training or not.
    """
    
    def __init__(self, visitor: NodeVisitor, strat: ValuePropagationStrategy, 
                 simulations: int, is_training: bool):
        """Initialize `Agent` instance.

        Args:
            visitor (NodeVisitor): Node visitor.
            strat (ValuePropagationStrategy): Value propagation strategy.
            simulations (int): The number of simulations conducted per one  
                action.
            is_training (bool): Value that indicates whether this instance is  
                training or not.
        """
        self._history = []
        self._visitor = visitor
        self._strat = strat
        self._simulations = simulations
        self._is_training = is_training
        
    def act(self, o: Observation) -> Action:
        """Act an appropriate action on the given observation.  
        
        The appropriate action is calculated by using MCTS.
        
        Args:
            o (Observation): The observation.
        
        Returns:
            Action: The appropriate action.
        """
    
    def add_reward(self, r: Reward) -> None:
        """Add the given reward to the history of this instance.
        
        Args:
            r (Reward): The reward.
        """ 

    def prop_v(self) -> None: 
        """Propagate state values of states that this instance visited.
        """
    
    def clear_history(self) -> None: 
        """Clear the history of this instance.
        """

    def update(self, model: Model) -> None:
        """Update this instance with the given model.

        Args:
            model (Model): The neural networks model.
        """

    def get_history(self) -> list: 
        return self._history

    def train(self, 
              actors: int, 
              simulations: int = 800, 
              alpha: float = 1.0,
              epsilon: float = 0.25, 
              window_size: int = 1e6, 
              steps: int = 700e3, 
              epochs: int = 1, 
              batch_size: int = 4096, 
              mini_batch_size: int = 32, 
              save_steps: int = 1e3, 
              save_path: str = "bin/alphazero.keras") -> None:
        """Train `model` of this instnace with `epochs` epochs for `steps`   
        steps.  
        
        `actors` threads generate data and main thread trains `model` of   
        this instance with the generated data. Data generation and model  
        training are processed parellely.
                
        Args:
            actors (int): The number of threads for data generation.
            simulations (int, optional): The number of mcts simulations   
                (default `800`).
            alpha (float, optional): Dirichlet noise parameter. If alpha is   
                smaller than `1.0`, the distribution vectors become near the   
                basis vector. If alpha is `1.0`, the distribution vectors are   
                uniform. If alpha is bigger than `1.0`, the distribution   
                vectors become more-balanced. Alpha is preferred to be an   
                inverse proportion to the approximate number of legal actions   
                (default `1.0`).
            epsilon (float, optional): Weight of dirichlet noise   
                (default `0.25`).
            window_size (int, optional): The number of episodes to be saved in  
                replay buffer (default `1e6`). 
            steps (int, optional): The number of steps to train  
                (default `700e3`).
            epochs (int, optional): The number of epochs to train in one step.  
                (default `1`).
            batch_size (int, optional): The number of data sampled from the   
                replay buffer for training (default `4096`).
            mini_batch_size (int, optional): Mini batch size of data for   
                training (defaults `32`).
            save_steps (int, optional): The number of steps for saving trained     
                model (default `1e3`). 
            save_path (str, optional): File path for saving trained model   
                (default `bin/alphazero.tf`).
        """
        model = Node.get_model()
        simulator = Node.get_simulator()
        replay_mem = ReplayMemory(replay_buffer_size)
        rng = np.random.default_rng()
        gamma = Node.get_gamma()
        type = Node.get_type()

        for i in range(1, steps + 1):
            for j in range((i - 1) * episodes + 1, i * episodes + 1):
                episode_buffer = []
                episode_rewards = []
                
                s = simulator.gen_init_s()
                p, v = model(s[np.newaxis, :], False)
                p = ((1 - epsilon) * p 
                     + epsilon * rng.dirichlet([alpha for _ in range(p.shape[0])]))
                
                root = Node(s, p, False)

                while 1:
                    pi = root.mcts(simulations)
                    a = rng.choice(pi.shape[0], p=pi)
                    
                    episode_buffer.append([j, root.get_s(), pi])
                    episode_rewards.append(root.get_reward(a))
                    
                    root = root.get_child(a)
                    
                    if root.get_is_terminal():
                        break
                
                traj_len = len(episode_buffer)
                
                episode_buffer[traj_len - 1].append(episode_rewards[traj_len - 1])
                
                for k in range(traj_len - 2, -1, -1):
                    v = episode_buffer[k + 1][3] if type == 0 \
                            else -episode_buffer[k + 1][3]
                    episode_buffer[k].append(episode_rewards[k] + gamma * v)
        
                replay_mem.extend(episode_buffer)

            x, y = replay_mem.sample(examples)
            model.fit(x, y, epochs=epochs, batch_size=mini_batch_size)
            
            if i % save_steps == 0:
                model.save(save_path)    
                