"""AlphaZero algorithm and their variants."""

from threading import Thread

import keras
import tensorflow as tf

from alphazero.components.agent import Agent
from alphazero.components.environment import Environment
from alphazero.improvements.policy_improvements import PVCD
from alphazero.improvements.value_estimations.simulation_value_estimations \
        import SoftZ
from alphazero.improvements.value_estimations.trajectory_value_estimations import FinalOutcome
from alphazero.mdp.factory import MDPFactory
from alphazero.nn.model import Model
from alphazero.tree.visitor import BoundedVisitor, NodeVisitor, PUCTSelector
from alphazero.utils.replaybuffer import ReplayBuffer


class BaseAlphaZero():
    """Base AlphaZero class where training and inference logic alives.

    Note:
        - Descendants of this class that overrides `__init__()` should call  
          `BaseAlphaZero.__init__()` in its `__init__()`.
        - Direct descendants of this class should override `create_agent()`.

    Attributes:
        _model (Model): Latest neural networks model.
        _replay_buffer (ReplayBuffer): Replay buffer.
        _training_step (int): Current training step.
        _done_training (bool): Flag that indicates whether training is done  
            or not.
    """
       
    def __init__(self, factory_cls: type, model: Model | str,  
                 capacity: int, is_multiagent: bool):
        """Initialize `BaseAlphaZero` instance.

        Args:
            factory_cls (type): MDP factory class.
            model (Model or str): Neural networks model or file name of a  
                saved model.
            capacity (int): Capacity of replay buffer.
            is_multiagent (bool): Value that indicates whether the problem is  
                multiagent or not.
        """ 
        keras.saving.register_keras_serializable('az.nn', 'Model')(Model)
        keras.saving.register_keras_serializable()(factory_cls)
                   
        self._model = model if isinstance(model, Model) \
                            else keras.models.load_model(model)
        self._replay_buffer = ReplayBuffer(capacity)
        self._training_step = 1
        self._done_training = False
        
        NodeVisitor.set_is_multiagent(is_multiagent)
        
    def train(self, actors: int, train_steps: int, epochs: int, 
              batch_size: int, mini_batch_size: int, save_steps: int, 
              simulations: int, preserve: bool) -> None:
        """Train neural networks model of this instance.

        Args:
            actors (int): The number of actors that generate data.
            train_steps (int): The number of training steps.
            epochs (int): The number of epochs per one training step. 
            batch_size (int): The number of training samples per one epoch.
            mini_batch_size (int): Mini-batch size.
            save_steps (int): The number of steps for saving trained model.
            simulations (int): The number of MCTS simulations.
            preserve (bool): Flag that indicates preserving tree statistics  
                after an action.
        """
        threads = []
        
        for _ in range(actors):
            t = Thread(target=self._gen_data, args=(simulations, preserve))
            t.start()
            threads.append(t)
        
        t = Thread(target=self._train_model, 
                   args=(train_steps, epochs, batch_size, 
                         mini_batch_size, save_steps))
        t.start()
        threads.append(t)
        
        for t in threads:
            t.join()
        
    def _gen_data(self, simulations: int, preserve: bool) -> None:
        """Generate episode data and save them to the replay buffer of this  
        instance.

        Args:
            simulations (int): The number of MCTS simulations.
            preserve (bool): Flag that indicates preserving tree statistics  
                after an action.
        """
        env = self.create_env(self._model.get_factory())
        agent = self.create_agent(self._model)
        
        while not self._done_training:
            while not env.is_terminated():
                env.apply(agent.act(env.get_o(), simulations, preserve))
                agent.add_r(env.get_r())
            
            self._replay_buffer.add_history(agent.obtain_history())
            
            env.reset()
            agent.reset()
            agent.update(self._model, self._training_step)
    
    def create_env(self, factory: MDPFactory) -> Environment:
        """Create an appropriate `Environment` instance using given MDP  
        factory.

        Args:
            factory (MDPFactory): The MDP factory.
            
        Returns:
            Environment: The `Environment` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override'
                                   'create_env().')
    
    def create_agent(self, model: Model) -> Agent:
        """Create an appropriate `Agent` instance using given neural networks  
        model.

        Args:
            model (Model): The neural networks model.

        Returns:
            Agent: The `Agent` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override' 
                                   'create_agent().')

    def _train_model(self, train_steps: int, epochs: int, batch_size: int,
                      mini_batch_size: int, save_steps: int) -> None:
        """Train the neural networks model of this instance.

        Args:
            train_steps (int): The number of training steps.
            epochs (int): The number of epochs per one training step.
            batch_size (int): The number of training samples per one epoch.
            mini_batch_size (int): Mini-batch size.  
            save_steps (int): The number of steps for saving trained model.
        """
        while self._training_step <= train_steps:
            model = keras.models.clone_model(self._model)
            model.save_weights(self._model.get_weights())
            
            x, y = self._replay_buffer.sample(batch_size)
            model.fit(x, y, batch_size=mini_batch_size, epochs=epochs)
            
            if not self._training_step % save_steps:
                model.save(f'{self._training_step}.keras')
            
            self._model = model        
            self._training_step += 1
        
        self._done_training = True
        
    def infer(self, preserve: bool) -> None:
        """Infer a solution of the problem.
        
        Args: 
            preserve (bool): Flag that indicates whether to preserve the  
                subtree or not after an action.
        """


class AlphaZero(BaseAlphaZero):
    """AlphaZero algorithm.

    Note:
        - Descendants of this class that overrides `__init__()` should call  
          `AlphaZero.__init__()` in its `__init__()`.
    
    Attributes:
        _concentration (float): Concentration parameter.
        _noise_weight (float): Noise weight.

    Examples:
        The example of applying AlphaZero algorithm to user specific domain  
        is as following:
    
        ```python
        class MyAlphaZero(AlphaZero):
            
            def create_env(self, factory):
                return MyEnvironment(factory)
        
        #instantiates your AlphaZero.
        az = MyAlphaZero(...)
        
        #training.
        az.train(...)
        
        #inference.
        az.infer(...)
        ```
    """
    
    def __init__(self, factory_cls: type, model: Model | str, 
                 capacity: int, concentration: float, 
                 noise_weight: float, is_multiagent: bool):
        """Initialize `AlphaZero` instance.
        
        Args:
            factory_cls (type): MDP factory class.
            model (Model or str): Neural networks model or file name of a  
                saved model.
            capacity (int): Capacity of replay buffer.
            concentration: Concentration parameter of Dirichlet distribution.
            noise_weight: Dirichlet noise weight.
            is_multiagent (bool): Value that indicates whether the problem is  
                multiagent or not.
        """
        super(AlphaZero, self).__init__(factory_cls, model, 
                                        capacity, is_multiagent)
       
        self._concentration = concentration
        self._noise_weight = noise_weight
        
    def create_agent(self, model: Model) -> Agent:
        """Create AlphaZero agent.

        Note: 
            - This method overrides `BaseAlphaZero.create_agent()`.
        """
        return Agent(BoundedVisitor(model, 
                                    PUCTSelector(self._concentration, 
                                                 self._noise_weight)),
                     PVCD(),
                     SoftZ(),
                     FinalOutcome())
