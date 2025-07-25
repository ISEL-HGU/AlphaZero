"""Trajectory value estimation interface and algorithms."""


class TrajectoryValueEstimation():
    """Base trajectory dimension value estimation.
    
    Note:
        - Descendants of `TrajectoryValueEstimation` should override 
          `__call__()`.
    """
    
    def __call__(self, history: list[dict[str, object]]) -> None:
        """Evaluate all the state value in the given history.
        
        Args:
            history (list): The history.
        """
        raise NotImplementedError(f'class {self.__class__} did not override'
                                   'eval_all().')


class FinalOutcome(TrajectoryValueEstimation):
    """Final outcome trajectory dimension value estimation algorithm."
    """
    
    def __call__(self, history: list[dict[str, object]]) -> None:
        """Evaluate state value of the given history with the final outcome.
    
        Note:
            - This method overrides `TrajectoryValueEstimation.__call__()`.
        """
        for data in history:
            data['z'] = history[-1]['u']
            data.pop('v')