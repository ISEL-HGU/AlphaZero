import numpy as np

class Environment():
    """Interface that represents environment.
    This method is used for applying model of agent.
    Class that implements this interface should implement 
    `gen_init_s()`, `response()`, and `is_terminal()`.
    """
    
    def gen_init_s(self, *args, **kwargs):
        """Generates initial state. 

        NotImplementedError: Raises when class that implemented 
                this interface did not implement this method.
        
        Returns:
            np.ndarray: The initial state.
        """

        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method gen_init_s()')
    
    
    def response(self, s: np.ndarray, a: int):
        """Responses current state with selected action 
        by using given `s` and `a`.

        Args:
            s (np.ndarray): Current state.
            a (int): Selected action.
        
        Raises:
            NotImplementedError: Raises when class that implemented
                this interface did not implement this method.
        
        Returns:
            np.ndarray: Next state.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method response()')
    
    
    def is_terminal(self, s: np.ndarray) -> int:
        """Checks whether given state `s` is terminal state or not.

        Args:
            s (np.ndarray): The state to be checked.

        Returns:
            int: non-zero if the given state is terminal state, zero otherwise. 
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + "did not implement method is_terminal()") 
        