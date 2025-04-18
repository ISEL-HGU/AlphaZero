import numpy as np

class Simulator():
    """Alphazero simulator interface.  
    
    Class that implements this interface should implement 
    `simulate()`.
    """
        
    def simulate(self, s: np.ndarray, a: int) -> tuple:
        """Simulates taking action `a` on state `s`.
        
        Args:
            s (np.ndarray): The state to be applied the action.
            a (int): The action to be taken.
                
        Returns:
            tuple: Tuple that consists of immediate reward and next state.\\
                The next state should be `None` if `a` is not a valid action.

        Raises:
            NotImplementedError: Raises when class that implemented
                this interface did not implement this method.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method simulate()')
        