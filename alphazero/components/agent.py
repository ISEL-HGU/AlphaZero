"""Agent."""

import tensorflow as tf

from alphazero.algorithms.policy_improvements.policy_improvement \
        import PolicyImprovement
from alphazero.algorithms.value_estimation.tree.tree_value_estimation \
        import TreeValueEstimation
from alphazero.alphazero import BaseAlphaZero
from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation
from alphazero.nn.model import Model
from alphazero.tree.node import Node
from alphazero.tree.visitor import NodeVisitor

class Agent():
    """Agent that acts on an environment.

    Attributes:
        _history (list): History of the episode.
        _root (Node): Root node of search tree.
        _visitor (NodeVisitor): Node visitor.  
        _simulations (int): The number of simulations.
        _preserve (bool): Flag that indicates preserving the subtree.
    """
    
    def __init__(self, pi: PolicyImprovement, tve: TreeValueEstimation, 
                 discount_factor: float, visitor: NodeVisitor, 
                 simulations: int, preserve: bool):
        """Initialize `Agent` instance.

        Args:
            pi (PolicyImprovement): Pollicy improvement algorithm.
            tve (TreeValueEstimation): Tree demension value estimation  
                algorithm.
            discount_factor (float): Discount factor.
            visitor (NodeVisitor): Node visitor.
            simulations (int): The number of simulations for mcts.
            preserve (bool): Flag that indicates whether preserving the  
                subtree or not.
        """
        self._history = []
        self._root = Node.initial_root(pi, tve, discount_factor)
        self._visitor = visitor
        self._simulations = simulations
        self._preserve = preserve
        
    def act(self, o: Observation) -> Action:
        """Act an appropriate action on the given observation.

        The appropriate action is calculated by using MCTS.
        
        Args:
            o (Observation): The observation.
        
        Returns:
            Action: The appropriate action.
        """
        data = {}
        
        data['o'] = o
        
        pi, v = self._root.mcts(o, self._simulations, self._visitor)
        data['pi'] = pi
        data['v'] = v
        
        a = tf.random.categorical(tf.math.log(pi), 1).numpy()
        data['a'] = a
        
        self._history.append(data)
        
        next_root = self._root.get_child(a) 
        
        if not self._preserve:
            next_root.reset()
        
        next_root.make_root(self._root.get_pi(), self._root.get_tve())
        self._root = next_root
        
        return BaseAlphaZero.factory.create_action(a)

    def add_r(self, r: Reward) -> None:
        """Add the given reward to the history of this instance.

        Args:
            r (Reward): The reward.
        """ 
        self._history[-1]['u'] = r
    
    def clear_history(self) -> None: 
        """Clear the history of this instance.
        """
        self._history = []

    def update(self, model: Model, training_step: int) -> None:
        """Update this instance.

        Args:
            model (Model): The neural networks model.
            training_step (int): Current training_step.
        """
        self._visitor.set_model(model)

    def get_history(self) -> list: 
        return self._history
