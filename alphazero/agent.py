"""Module that trains nerual network.

Classes:
    Agent: Alphazero agent interface.
    Node: Mcts tree node.
"""

import numpy as np
import tensorflow as tf
from node import Node
from replaymemory import ReplayMemory
from alphazero.forwardables.simulator import Simulator

class Agent():
    """Alphazero agent interface.  
    
    Class that implement this interface should implement `_create_simulator()`.
    
    Attributes:
        model (tf.keras.Model): Neural network model.  
        train (function): Train `model`.
        infer (function): Infer using `model`. 
    """
    
    def __init__(self, model: tf.keras.Model, type: int = 0):
        """Initialize `Agent` instance.

        Args:
            model (tf.keras.Model): Neural network model.
            type (int, optional): Type of agent. `0` indicates single and `1`   
                indicates double agent. (default `0`)
        """

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
                
    def _gen_data(self, buffer: ReplayBuffer, simulations: int, 
                  alpha: float, epsilon: float) -> None:
        """Generate data by self-play.
        
        Args:
            buffer (ReplayBuffer): Replay buffer where generated data to be  
                saved.
            simulations (int): The number of mcts simulations.
            alpha: Dirichlet noise parameter.
            epsilon: Weight of dirichlet noise.
        """
        
    def _create_simulator(self) -> Simulator: 
        """Create concrete `Simulator` instance.

        Returns:
            Simulator: Simulator.
        """
         
    def infer(self, simulations: int) -> list[Action]:
        """Infer using `model` of this instance. 
        
        Args:
            simulations (int): The number of mcts simulations.
        
        Returns:
            list[Action]: List of actions conducted.         
        """
        