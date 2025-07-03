"""Final outcome value esetimation for trajectory demension."""

from alphazero.algorithms.value_estimation.trajectory.trajectory_value_estimation \
        import TrajectoryValueEstimation

class FinalOutcome(TrajectoryValueEstimation):
    """Final outcome trajectory value estimation."
    """
    
    def eval_all(self, history: list) -> None:
        """Evaluate state value of the given history with the final outcome.
        
        Note:
            This method overrides `eval_all()` of `TrajectoryValueEstimation`.
        """
        for data in history:
            data['z'] = history[-1]['u']
