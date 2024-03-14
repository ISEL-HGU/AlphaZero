import numpy as np
import tensorflow as tf
from environment import Environment
from node import Node
from replaymemory import ReplayMemory
from simulator import Simulator

class Agent():
    """Alphazero agent that trains and applies to the task.
    
    Args: 
        model (tf.keras.Model): Neural network model.
        simulator (Simulator): Simulator.
        type(int): Type of agent. `0` indicates using single agented mcts 
            and `1` indicates using multi agented mcts.
    """
    
    def __init__(self, model: tf.keras.Model, simulator: Simulator, type: int):
        Node.configure(model, simulator, type)

    
    def train(self, steps: int=700000, epochs: int=1,
              alpha: float=1.0, epsilon: float=0.25,
              episodes: int=100, replay_episodes: int=50000,
              simulations: int=800, 
              total_batch_size: int=4096, mini_batch_size: int=32, 
              save_steps: int=1000, save_path: str='bin/alphazero.tf') -> None:
        """Trains model of agent `epochs` epochs for `steps` steps. 
        The agent self-plays `episodes` episodes using mcts and trains 
        using generated episode data.
        
        Probabilities of root node is calculated as 
        `(1 - epsilon) * p_a + epsilon * eta_a`, where:
        * `p_a` are prior probabilities of the root node.
        * `eta_a` are dirichlet noise to the root node.
        
        Args: 
            steps (int, optional): The number of steps to train. 
                Setted to `700000` by default.
            epochs (int, optional): The number of epochs to train in one step.
                Setted to `1` by default.
            alpha (float, optional): Dirichlet noise parameter. 
                If alpha is smaller than `1.0`, the distribution vectors 
                become near the basis vector. If alpha is `1.0`, 
                the distribution vectors are uniform. 
                If alpha is bigger than `1.0`, the distribution vectors
                become more-balanced. Alpha is preferred to be 
                an inverse proportion to the approximate number of 
                legal actions. Setted to `1.0` by default.
            epsilon (float, optional): Weight of dirichlet noise. 
                Setted to `0.25` by default.  
            episodes (int, opitional): The number of episodes 
                that the agent progresses before training in one step. 
                Setted to `100` by default.
            replay_episodes (int, optional): The number of episodes to be saved 
                in replay memory. Setted to `50000` by default.
            simulations (int, optional): The number of mcts simulations 
                conducted. Setted to `800` by default.
            total_batch_size (int, optional): Total batch size of data 
                for training. Setted to `4096` by default.
            mini_batch_size (int, optional): Mini batch size of data 
                for training. Setted to `32` by default.
            save_steps (int, optional): The number of steps 
                for saving the model. Setted to `1000` by default. 
            save_path (str, optional): File path for saving model. 
                Setted to `bin/alphazero.tf` by default.
        """
        
        model = Node.get_model()
        simulator = Node.get_simulator()
        replay_mem = ReplayMemory(replay_episodes)
        rng = np.random.default_rng()
        type = Node.get_type()

        for i in range(1, steps + 1):
            for j in range((i - 1) * episodes + 1, i * episodes + 1):
                episode_buffer = []
                
                s = simulator.gen_init_s()
                p, v = model(s[np.newaxis, :], False)
                p = ((1 - epsilon) * p 
                     + epsilon * rng.dirichlet([alpha for _ in range(p.shape[0])]))
                
                root = Node(s, p, False)

                while 1:
                    pi = root.mcts(simulations)
                    
                    episode_buffer.append([j, root.get_s(), pi])
                    
                    root = root.get_child(rng.choice(pi.shape[0], p=pi))
                    
                    if root.get_is_terminal():
                        break
                
                z = root.get_v_s()
                
                if type == 0:
                    for data in episode_buffer:
                        data.append(z)
                else:
                    if len(episode_buffer) % 2 != 0:
                        z = -z
                    
                    for data in episode_buffer:
                        data.append(z)
                        z = -z
                
                replay_mem.extend(episode_buffer)

            for _ in range(epochs):
                x, y = replay_mem.sample(total_batch_size)
                model.fit(x, y, batch_size=mini_batch_size)
            
            if i % save_steps == 0:
                model.save(save_path)    
            
                                   
    def apply(self, env: Environment, *args, **kwargs) -> None:
        """Applies this instance to the task.
        
        Args: 
            env (Environment): Environment of the task.
        """
