"""Test cases for AlphaZero and its variants."""

import unittest.mock

import keras
import tensorflow as tf

from alphazero.alphazero import BaseAlphaZero
from alphazero.components.agent import Agent
from alphazero.components.environment import Environment
from alphazero.mdp.factory import MDPFactory
from alphazero.nn.model import Model


class TestBaseAlphaZero(unittest.TestCase):
    """Class that tests functionalities of `BaseAlphaZero`.
    """
    
    @classmethod
    def setUpClass(cls):
        cls._model = unittest.mock.Mock(Model)
        cls._factory = unittest.mock.Mock(MDPFactory)
        cls._capacity = 10
    
    def test_train(self):
        """Test `BaseAlphaZero.train()`.
        
        This method checks 1 case:
        - A base alphazero algorithm with hyperparameters.
        
        Test case 1  
        **Given** a base alphazero algorithm with hyperparameters  
        **When** the algorithm trains the model  
        **Then** the model should be cloned and all threads should be  
        successfully terminated.
        """ 
        dummy_az = DummyBaseAlphaZero(TestBaseAlphaZero._model, 
                                      TestBaseAlphaZero._factory, 
                                      TestBaseAlphaZero._capacity, 
                                      False, 
                                      'DummyAlphaZero')
        actors = 3
        train_steps = 10
        epochs = 1
        batch_size = 10
        mini_batch_size = 5
        save_steps = 5
        simulations = 10
        preserve = False
        
        dummy_az.train(actors, train_steps, epochs, batch_size, 
                       mini_batch_size, save_steps, simulations, preserve)
        
        TestBaseAlphaZero._model.assert_not_called()
    
    
class DummyBaseAlphaZero(BaseAlphaZero):
    def __init__(self, model, factory, capacity, is_multiagent, name):
        super(DummyBaseAlphaZero, self).__init__(model, factory, capacity,
                                                 is_multiagent, name)
    
    def _create_agent(self, model: Model, is_multiagent: bool) -> Agent:
        def mock__len__(self):
            return 10
        
        def mock__getitem__(self, key):
            return {'o': [], 'a': [], 'pi': [], 'z': [], 'u': []}
                
        history = unittest.mock.Mock(list)
        history.__len__ = mock__len__
        history.__getitem__ =  mock__getitem__
        
        return unittest.mock.Mock(Agent, 
                                  **{'obtain_history.return_value': history})
    
    def _create_env(self) -> Environment:
        return unittest.mock.Mock(
                Environment, **{'is_terminated.return_value': True})
    
    def _create_model(self):
        return unittest.mock.Mock(Model)
