"""Interface Forwardable.
"""

from alphazero.mdp import State, Action

class Forwardable():
    """Interface that indicates a class can process one step.
    
    Class that inherits this class should implement `forward()`.
    """
    
    def forward(self, s: State, a: Action) -> tuple:
        return NotImplementedError(f'class {self.__class__.__name__} did not \
                                   implement forward().')
        