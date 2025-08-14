"""MDP state and its observation interface."""

import tensorflow as tf

class Observation():
    """Base observation class.

    Note:
        - `Observation` should be subclassed.
        - Descendants of `Observation` should call `Observation.__init__()` in
          their constructor.
        - Descendants of `Observation` should override `is_terminal()` and
          `get_features()`.
    
    Attributes:
        _repr (tf.Tensor): Observation representation.
    
    Examples:
        We recommend descendants of `Observation` to make the representation  
        of an observation in their constructor. Here is the example of 4 x 4  
        grid world's observation.
        
        ```python
        class GridWorldObservation(Observation):
            
            def __init__(self, x: int, y: int):
                # Makes representation of the observation.
                repr = tf.zeros([4, 4], tf.int32)
                repr[-y + 3][x] = 1
                
                super(GridWorldObservation, self).__init__(repr)
                
                self._g = (3, 3)
                self._x = x
                self._y = y
            
            def is_terminal(self) -> bool:
                return (self._x, self._y) == self._g
            
            def get_features(self) -> tuple[object]:
                return (self._x, self._y)
        ``` 
    """

    def __init__(self, repr: tf.Tensor):
        """Initialize `Observation` instance.

        Args:
            val (tf.Tensor): Observation representation.
        """
        self._repr = repr

    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal observation or not.

        Note:
            - This method will be called by `Environment.is_terminated()` and
              determine the episode termination.

        See Also:
            :py:meth:`~environment.Environment.is_terminated`

        Returns:
            bool: `True` if this instance represents terminal observation,  
                `False` otherwise.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                   'is_terminal().')
    
    def get_features(self) -> tuple[object]:
        """Gets features of this instance.

        Features denote all the attributes that are needed for processing to  
        the next state and creating a state.
        
        Returns:
            tuple: The features.
        """
        raise NotImplementedError(f'class {self.__class__} did not override '
                                   'get_features().')
        
    def get_repr(self) -> tf.Tensor:
        return self._repr


class State():
    """Base state class.  

    This class supports unbounded simulation.
    
    Note: 
        - `State` can be subclassed for custom simulation termination.
        - Descendants of `State` should call `State.__init__()` in their
          constructor.
        - Descendants of `State` should override `is_terminal()` and
          `get_features()`.

    Attributes:
        _repr (tf.Tensor): State representation.
    
    Examples:
        We recommend descendants of this class to set features from the  
        representation in their constructor. Here is the example of 4 x 4  
        grid world state.
        
        ```python
        class GridWorldState(State):
        
            def __init__(self, repr: tf.Tensor, x: int, y: int):
                super(GridWorldState, self).__init__(repr)
                
                self._g = (3, 3)
                self._x = x
                self._y = y 
    
            def is_terminal(self) -> bool:
                return (self._x, self._y) == self._g
        ```
    """
    
    def __init__(self, repr: tf.Tensor):
        """Initialize `State` instance.

        Args:
            repr (tf.Tensor): State representation.
        """
        self._repr = repr
    
    def is_terminal(self) -> bool:
        """Check whether this instance represents terminal state or not.

        Note: 
            - This method will be called by `NodeVisitor.expand_and_eval()` and
              determine the simulation termination.

        See Also: 
            :py:meth:`~visitors.NodeVisitor.expand_and_eval`
        
        Returns:
            bool: `True` if this instance represents terminal state, `False`  
                otherwise.
        """
        return False
    
    def get_repr(self) -> tf.Tensor:
        return self._repr    
