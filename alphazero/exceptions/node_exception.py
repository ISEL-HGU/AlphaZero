"""Node exception and its exception code."""

import enum


class NodeExceptionCode(enum.Enum):
    NONROOT = 1
    EXPANDED = 2
    UNEXPANDED = 3
    UNDETERMINED = 4

    def __str__(self):
        return self.name.swapcase()


class NodeException(Exception):
    """Exception that is raised when an unappropriate node calls an illegal  
    method.
    
    Attributes:
        _exc_code (NodeExceptionCode): Node exception code.
        _msg (str): Exception message.
    """

    def __init__(self, mqn: str, exc_code: NodeExceptionCode):
        """Initialize `NodeException` instance.

        Args:
            mqn (str): Qualified name of the method that raises this instance.
            exc_code (NodeExceptionCode): Node exception code. 
        """
        self._exc_code = exc_code
        self._msg = f'cannot call {mqn} if the node is {exc_code}.'       

    def get_exc_code(self):
        return self._exc_code