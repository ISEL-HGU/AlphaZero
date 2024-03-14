import numpy as np
from collections import deque

class ReplayMemory(deque):
    """Replay memory that saves data generated from each episode.
    
    Args:
        ep_cap (int): The number of episodes can be saved.
    """    
    
    #instance variables
    _ep_cap: int
    _min_ep: int
    _rng: np.random.Generator
    
    def __init__(self, ep_cap: int):
        super(ReplayMemory, self).__init__()

        self._ep_cap = ep_cap
        self._min_ep = 0
        self._rng = np.random.default_rng()
    
    
    def extend(self, data: list) -> None:
        """Saves `data` to this instance.
        
        All the data from minimum episode number are discarded 
            if this instance already has `ep_cap` numbers of episodes.
        
        Args:
            data (list): The data to be saved. The data should be a
                list of episode number, s, pi, and z, where:
            * `s` indicates state
            * `pi` indicates policy from mcts
            * `z` indicates real result value
        """
        
        if self._min_ep == 0:
            self._min_ep += 1
        
        if data[0][0] > self._min_ep + self._ep_cap - 1:
            while 1:
                if len(self) == 0:
                    break
                
                x = self.popleft()
                
                if x[0] > self._min_ep:
                    self.appendleft(x)
                    
                    break
            
            self._min_ep += 1
        
        super(ReplayMemory, self).extend(data)
        
     
    def sample(self, n: int) -> tuple:
        """Samples random `n` data from this instance.

        Args:
            n (int): The number of data to be sampled.
        
        Returns:
            tuple: The sampled data. The tuple is consist of 
                x and list of pi and z.
        """

        x = []
        y_pi = []
        y_z = []
        
        for i in self._rng.choice(len(self), size=n, replace=False):
            x.append(self[i][1])
            y_pi.append(self[i][2])
            y_z.append([self[i][3]])
            
        return (np.array(x), [np.array(y_pi), np.array(y_z)])  
        