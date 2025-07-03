"""Base value estimation for trajectory demension."""

class TrajectoryValueEstimation():
    """Base trajectory value estimation.
    
    Note:
        Descendants of `TrajectoryValueEstimation` should override `eval_all()`.
    """
    
    def eval_all(self, history: list) -> None:
        """Evaluate all the state value in the given history.
        
        Args:
            history (list): The history.
        """
        raise NotImplementedError(f'class {self.__class__} did not override \
                                  eval_all().')
