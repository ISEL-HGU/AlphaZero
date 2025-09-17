"""AlphaZero algorithm and their variants."""

from threading import Thread

import keras

from alphazero import mdp
from alphazero.components.agent import Agent
from alphazero.components.environment import Environment
from alphazero.improvements.policy_improvements import PVCD
from alphazero.improvements.value_estimations.simulation_value_estimations \
        import SoftZ
from alphazero.improvements.value_estimations.trajectory_value_estimations \
        import FinalOutcome
from alphazero.mdp.factory import MDPFactory
from alphazero.nn.model import Model
from alphazero.tree.visitor import PUCTVisitor
from alphazero.utils.replaybuffer import ReplayBuffer


class BaseAlphaZero():
    """Base AlphaZero class where training and inference logic alive.

    Note:
        - Descendants of this class should call `BaseAlphaZero.__init__()` in
          their `__init__()`.
        - Descendants of this class should override
          `BaseAlphaZero._create_agent()`.

    Attributes:
        _model (Model): Latest neural networks model.
        _replay_buffer (ReplayBuffer): Replay buffer.
        _training_step (int): Current training step.
        _done_training (bool): Flag of training status.
        _is_mulitagent (bool): Flag of multi-agent problem setting.
    """
       
    def __init__(self, model: Model | str, factory: MDPFactory,  
                 capacity: int, is_multiagent: bool, name: str):
        """Initialize `BaseAlphaZero` instance.

        Args:
            model (Model or str): Neural networks model or file name of a  
                saved model.
            factory (MDPFactory): MDP Factory.
            capacity (int): Capacity of replay buffer.
            is_multiagent (bool): Flag that indicates whether the problem is  
                in multi-agent setting or not.
            name (str): The name of the algorithm.
        """ 
        mdp.factory = factory
        
        self._model = model if isinstance(model, Model) \
                            else keras.models.load_model(model)
        self._replay_buffer = ReplayBuffer(capacity)
        self._training_step = 1
        self._done_training = False
        self._is_multiagent = is_multiagent
        self._name = name
    
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
        env = self._create_env()
        agent = self._create_agent(self._model, self._is_multiagent)
        
        while not self._done_training:
            while True:
                a = agent.act(env.get_r(), env.get_o(), simulations, preserve)
                
                if env.is_terminated():
                    break
                
                env.apply(a)
                
            self._replay_buffer.add_history(agent.obtain_history())
            
            env.reset()
            agent.reset()
            agent.update(self._model, self._training_step)
    
    def _create_env(self) -> Environment:
        """Create an appropriate `Environment` instance.
            
        Returns:
            Environment: The `Environment` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                  '_create_env().')
    
    def _create_agent(self, model: Model, is_multiagent: bool) -> Agent:
        """Create an appropriate `Agent` instance using given neural networks  
        model.

        Args:
            model (Model): The neural networks model.
            is_multiagent (bool): Flag that indicates whether the agent to be  
                created is multi-agent or not.

        Returns:
            Agent: The `Agent` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                  '_create_agent().')
    
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
        while len(self._replay_buffer) < batch_size:
            pass
        
        while self._training_step <= train_steps:
            model = self._create_model()
            model.set_weights(self._model.get_weights())

            x, y = self._replay_buffer.sample(batch_size)
            model.fit(x, y, batch_size=mini_batch_size, epochs=epochs)
            
            if not self._training_step % save_steps:
                model.save(f'{self._training_step}.keras')
            
            self._model = model        
            self._training_step += 1
        
        self._done_training = True
        
    def _create_model(self) -> Model:
        """Create an appropriate `Model` instance.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                  '_create_model().')    
        
    def infer(self, preserve: bool) -> None:
        """Infer a solution of the problem.
        
        Args: 
            preserve (bool): Flag that indicates whether to preserve the  
                subtree or not after an action.
        """


class AlphaZero(BaseAlphaZero):
    """AlphaZero algorithm.

    Note:
        - Descendants of this class should call `AlphaZero.__init__()` in their
          `__init__()`.
        - Descendants of this class should override
          `BaseAlphaZero._create_env()` and `BaseAlphaZero._create_model()`.
    
    Attributes:
        _concentration (float): Concentration parameter.
        _noise_weight (float): Noise weight.

    Examples:
        The example of applying AlphaZero algorithm to user specific domain  
        is as following:
    
        ```python
        class MyAlphaZero(AlphaZero):
            
            def _create_env(self):
                return MyEnvironment()
            
            def _create_model(self):
                return Model(MyReprNet(), MyDynamics(), MyPredNet())
        
        # instantiates your AlphaZero.
        my_az = MyAlphaZero(...)
        
        # training.
        my_az.train(...)
        
        # inference.
        my_az.infer(...)
        ```
    """
    
    def __init__(self, model: Model | str, factory: MDPFactory, 
                 capacity: int, concentration: float, 
                 noise_weight: float, is_multiagent: bool):
        """Initialize `AlphaZero` instance.
        
        Args:
            model (Model or str): Neural networks model or file name of a  
                saved model.
            factory (MDPFactory): MDP factory.
            capacity (int): Capacity of replay buffer.
            concentration: Concentration parameter of Dirichlet distribution.
            noise_weight: Dirichlet noise weight.
            is_multiagent (bool): Value that indicates whether the problem is  
                multiagent or not.
        """
        super(AlphaZero, self).__init__(model, factory, capacity, 
                                        is_multiagent, 'AlphaZero')
       
        self._concentration = concentration
        self._noise_weight = noise_weight
        
    def _create_agent(self, model: Model, is_multiagent: bool) -> Agent:
        """Create AlphaZero agent.

        Note: 
            - This method overrides `BaseAlphaZero._create_agent()`.
        """
        return Agent(PUCTVisitor(model, self._concentration, 
                                 self._noise_weight, is_multiagent),
                     PVCD(),
                     SoftZ(),
                     FinalOutcome())
