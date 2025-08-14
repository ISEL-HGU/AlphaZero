"""Test cases for visitors."""

import unittest.mock

import tensorflow as tf

from alphazero import mdp
from alphazero.exceptions.node_exception \
		import NodeException, NodeExceptionCode
from alphazero.mdp.action import Action
from alphazero.mdp.factory import MDPFactory
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import State
from alphazero.nn.model import Model
from alphazero.tree.node import Node
from alphazero.tree.visitor import NodeVisitor
from tests.tree.test_node import TestNode

class TestNodeVisitor(unittest.TestCase):
	"""Class that tests functionalities of `NodeVisitor`.
	"""
	
	@classmethod
	def setUpClass(cls):
		cls.model_terminal = unittest.mock.Mock(
				Model,
				**{'encode.return_value': unittest.mock.Mock(
							State, **{'is_terminal.return_value': True}),
				   'process.return_value': 
		   					(unittest.mock.Mock(Reward),
							 unittest.mock.Mock(
									State, 
				  					**{'is_terminal.return_value': True})),
				   'estimate.return_value': (tf.constant([0.7, 0.3]), 0.5)})
		
		cls.model_nonterminal = unittest.mock.Mock(
				Model,
				**{'encode.return_value': unittest.mock.Mock(
							State, **{'is_terminal.return_value': False}),
				   'process.return_value': 
		   					(unittest.mock.Mock(Reward),
							 unittest.mock.Mock(
									State, 
							  		**{'is_terminal.return_value': False})),
				   'estimate.return_value': (tf.constant([0.7, 0.3]), 0.5)})		
  		
	def test_post_visit_internal(self):
		"""Test `NodeVisitor.post_visit_internal()`.
		
		This method test 3 cases:
		- A visitor with an unexpanded node.
		- A single-agent node visitor with an expanded node.
		- A multi-agent node visitor with an expanded node.
		
		Test case 1  
		**Given** a node visitor with an unexpanded node  
		**When** the visitor conducts post visit internal operation  
		**Then** `NodeException` of the code `NodeExceptionCode.UNEXPANDED`  
		should be thrown.
		
		Test case 2  
		**Given** a single-agent node visitor with an expanded node.  
		**When** the visitor conducts post visit internal operation  
		**Then** the state-action value of the nodeshould be returned.
		
		Test case 3  
		**Given** a multi-agent node visitor with an expanded node.  
		**when** the visitor conducts post visit internal operation  
		**Then** the state-action value of the node with the opposite sign    
		should be returned.
		"""
		arbitrary_visitor = NodeVisitor(TestNodeVisitor.model_nonterminal, 
								  		False)
		visitors = [NodeVisitor(TestNodeVisitor.model_nonterminal, False), 
					NodeVisitor(TestNodeVisitor.model_nonterminal, True)]
		unexpanded_node = unittest.mock.Mock(
	  			Node, **{'is_expanded.return_value': False})
		expanded_node = unittest.mock.Mock(
				Node,  
				**{'update_stats.side_effect': calc_q,
	   			   'is_expanded.return_value': True})
		v = 0.5
		V_expected = [0.85, -0.85]
		V_actual = []
		
		with self.assertRaises(NodeException) as cm:
			arbitrary_visitor.post_visit_internal(unexpanded_node, v)

		for visitor in visitors:
			V_actual.append(visitor.post_visit_internal(expanded_node, v))

		self.assertIs(cm.exception.get_exc_code(), 
				   	  NodeExceptionCode.UNEXPANDED)
 
		for v_actual, v_expected in zip(V_actual, V_expected):
			self.assertAlmostEqual(v_actual, v_expected)

	def test_visit_leaf(self):
		"""Test `NodeVisitor.test_visit_leaf()`.
		
		This method tests 11 cases:
		- A node visitor with an expanded node.
		- A single-agent node visitor with a terminal node.
		- A single-agent node visitor with an undetermined root node that  
  		  turns out to be a terminal.
		- A single-agent node visitor with an undetermined root node that  
  		  turns out to be a non-terminal.
		- A single-agent node visitor with an undetermined non-root node that  
  		  turns out to be a terminal. 
		- A single-agent node visitor with an undetermined non-root node that  
		  turns out to be a non-terminal.
		- A multi-agent node visitor with a terminal node.
		- A multi-agent node visitor with an undetermined root node that  
  		  turns out to be a terminal.
		- A multi-agent node visitor with an undetermined root node that  
  		  turns out to be a non-terminal.
		- A multi-agent node visitor with an undetermined non-root node that  
  		  turns out to be a terminal. 
		- A multi-agent node visitor with an undetermined non-root node that  
		  turns out to be a non-terminal.
	
		Test case 1  
		**Given** a node visitor with an expanded node  
		**When** the node visitor conducts visit leaf operation  
		**Then** the `NodeException` of the code `NodeExceptionCode.EXPANDED`  
		should be raised.
  
		Test case 2  
		**Given** a single-agent node visitor with a terminal node  
		**When** the node visitor conducts visit leaf operation  
		**Then** the calculated value should be returned.
  
		Test case 3  
		**Given** a single-agent node visitor with an undetermined root node  
  		that turns out to be a terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** `0` should be returned.
		
		Test case 4  
		**Given** a single-agent node visitor with an undetermined root node that  
  		turns out to be a non-terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** the value of the node's state transition should be returned.
  
		Test case 5  
		**Given** a single-agent node visitor with an undetermined non-root  
  		node that turns out to be a terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** the reward of the node should be returned.
		
		Test case 6  
		**Given** a single-agent node visitor with an undetermined non-root  
  		node that turns out to be a non-terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** the state-action value of the node should be returned.
		
		Test case 7  
		**Given** a multi-agent node visitor with a terminal node  
		**When** the node visitor conducts visit leaf operation  
		**Then** the calculated value with the opposite sign should be  
  		returned.
		
		Test case 8  
		**Given** a multi-agent node visitor with an undetermined root node  
  		that turns out to be a terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** `0` should be returned.  
		
		Test case 9  
		**Given** a multi-agent node visitor with an undetermined root node  
  		that turns out to be a non-terminal  
		**When** the node visitor conducts visit leaf operation  
		**Then** the value of the node's state transition with the opposite  
  		sign should be returned.
		
		Test case 10  
		**Given** a multi-agent node visitor with an undetermined non-root  
  		node that turns out to be a terminal    
		**When** the node visitor conducts visit leaf operation  
		**Then** the reward of the node with the opposite sign should be  
		returned.
		
		Test case 11  
		**Given** a multi-agent node visitor with an undetermined non-root  
  		node that turns out to be a non-terminal 
		**When** the node visitor conducts visit leaf operation  
		**Then** the state-action value of the node with the opposite sign  
  		should be returned.
		"""
		arbitrary_visitor = NodeVisitor(TestNodeVisitor.model_nonterminal, 
								  		False)
		visitors = [[NodeVisitor(TestNodeVisitor.model_terminal, False),
			  		 NodeVisitor(TestNodeVisitor.model_nonterminal, False)],
					[NodeVisitor(TestNodeVisitor.model_terminal, True),
					 NodeVisitor(TestNodeVisitor.model_nonterminal, True)]]
		expanded_node = unittest.mock.Mock(
	  			Node, **{'is_expanded.return_value': True})
		terminal_node = unittest.mock.Mock(
	  			Node,  
				**{'update_stats.side_effect': calc_q,
				   'is_expanded.return_value': False,
				   'is_terminal.return_value': True})
		
		undetermined_nodes = [
				unittest.mock.Mock(Node,
							 	   **{'update_stats.side_effect': calc_q,
									  'is_expanded.return_value': False,
									  'is_terminal.return_value': False,
									  'is_root.return_value': True}),
				unittest.mock.Mock(Node,
							 	   **{'update_stats.side_effect': calc_q,
									  'is_expanded.return_value': False,
									  'is_terminal.return_value': False,
									  'is_root.return_value': False})]
		V_expected = [0.4, 0.4, 0.85, 0.4, 0.85, -0.4, -0.4, 0.05, -0.4, 0.05]
		V_actual = []
		
		with self.assertRaises(NodeException) as cm:
			arbitrary_visitor.visit_leaf(expanded_node)
					
		for perspective_visitors in visitors:
			V_actual.append(perspective_visitors[0].visit_leaf(terminal_node))

			for undetermined_node in undetermined_nodes:
				for perspective_visitor in perspective_visitors:
					V_actual.append(
							perspective_visitor.visit_leaf(undetermined_node))

		self.assertIs(cm.exception.get_exc_code(), NodeExceptionCode.EXPANDED)
  
		for v_actual, v_expected in zip(V_actual, V_expected):
			self.assertAlmostEqual(v_actual, v_expected)


def setUpModule():
	mdp.factory = unittest.mock.Mock(
	 		MDPFactory, 
			**{'create_action.return_value': unittest.mock.Mock(Action)})

def calc_q(v: float) -> float:
	r = 0.4
	discount_factor = 0.9
	
	return r + discount_factor * v
