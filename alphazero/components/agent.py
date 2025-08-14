"""Agent."""

import tensorflow as tf

from alphazero import mdp
from alphazero.improvements.policy_improvements import PolicyImprovement
from alphazero.improvements.value_estimations.simulation_value_estimations \
        import SimulationValueEstimation
from alphazero.improvements.value_estimations.trajectory_value_estimations \
        import TrajectoryValueEstimation
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
        _pi (PolicyImprovement): Pollicy improvement algorithm.
        _sve (SimulationValueEstimation): Simulation value estimation algorithm.
        _tve (TrajectoryValueEstimation): Trajectory value estimation algorithm.
    """
    
    def __init__(self, visitor: NodeVisitor, 
                 pi: PolicyImprovement,
                 sve: SimulationValueEstimation, 
                 tve: TrajectoryValueEstimation):
        """Initialize `Agent` instance.

        Args:
            visitor (NodeVisitor): Node visitor.
            pi (PolicyImprovement): Pollicy improvement algorithm.
            sve (SimulationValueEstimation): Simulation dimension value  
                estimation algorithm.
            tve (TrajectoryValueEstimation): Trajectory dimension value  
                estimation algorithm.
        """
        self._history = []
        self._root = Node.root(pi, sve)
        self._visitor = visitor
        self._pi = pi
        self._sve = sve
        self._tve = tve
        
    def act(self, r: Reward | None, o: Observation | None, 
            simulations: int, preserve: bool) -> Action | None:
        """Act an appropriate action from given reward and observation.

        The appropriate action is calculated by using MCTS (Monte-Carlo Tree  
        Search).
        
        Args:
            r (Reward): The reward.
            o (Observation): The observation.
            simulations (int): The number of simulations for MCTS.
            preserve (bool): Flag that indicates whether to preserve the  
                subtree or not.
        
        Returns:
            Action: The appropriate action.
        """
        if r is not None:
            self._history[-1]['u'] = r.get_val()
        
        if o is None or o.is_terminal(): 
            return None
        
        data = {}
        
        data['o'] = o.get_repr()
        
        pi, v = self._root.mcts(o, simulations, self._visitor)
        data['pi'] = pi
        data['v'] = v
        
        a = mdp.factory.create_action(
                tf.random.categorical(tf.math.log(pi), 1).numpy())
        data['a'] = a.to_repr()
        
        self._history.append(data)
        
        next_root = self._root.get_child(a.get_num()) 
        
        if not preserve:
            next_root.reset()
        
        next_root.make_root(self._pi, self._sve)
        self._root = next_root
        
        return a
    
    def obtain_history(self) -> list[dict[str, object]]:
        """Obtain episode history of this instance.

        Trajectory dimension value estimation algorithm is applied to the  
        episode history of this instance.

        Returns:
            list: The episode history.
        """
        self._tve(self._history)
        
        return self._history
    
    def update(self, model: Model, training_step: int) -> None:
        """Update this instance.

        Update neural networks model and policy improvement algorithm of this  
        instance.
        
        Args:
            model (Model): The neural networks model.
            training_step (int): Current training_step.
        """
        self._visitor.set_model(model)
        self._pi.update_param(training_step)
    
    def reset(self) -> None:
        """Reset this instance.
        
        Clear history and reset root node and node visitor of this instance.
        """
        self._history = []
        self._root = Node.root(self._pi, self._sve)
        self._visitor.reset()
