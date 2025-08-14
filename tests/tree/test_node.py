"""Testcase for Node class."""

import unittest.mock

from alphazero.exceptions.node_exception \
        import NodeException, NodeExceptionCode
from alphazero.improvements.policy_improvements import PolicyImprovement
from alphazero.improvements.value_estimations.simulation_value_estimations \
        import SimulationValueEstimation
from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State
from alphazero.tree.node import Node
from alphazero.tree.visitor import NodeVisitor


class TestNode(unittest.TestCase):
    """Class that tests functionalities of `Node`.
    """
    
    @classmethod
    def setUpClass(cls) -> None:
        cls._intern_s = unittest.mock.Mock(State)
        cls._a = unittest.mock.Mock(Action)
        cls._p = 0.7
        cls._pi = unittest.mock.Mock(PolicyImprovement)
        cls._tve = unittest.mock.Mock(SimulationValueEstimation)
        cls._child = Node(unittest.mock.Mock(State), 
                          unittest.mock.Mock(Action),
                          1.0)
        
    def test_mcts(self) -> None:
        """Test `Node.mcts()`.
        
        This method checks 3 cases:
        - A non-root node.
        - An expanded root node.
        - An unexpanded root node.
        
        Test case 1  
        **Given** a non-root node  
        **When** the node conducts mcts  
        **Then** `NodeExcpetion` of the code `NodeException.NONROOT` should  
        be raised.
        
        Test case 2  
        **Given** an expanded root node and the number of simulations  
        **When** the node conducts mcts  
        **Then** the tree traversing should be done for the number of  
        simulations times.
        
        Test case 3  
        **Given** an unexpanded root node and the number of simulations  
        **When** the node conducts mcts  
        **Then** the tree traversing should be done for the number of  
        simulations+1 times.
        """
        nonroot = Node(TestNode._intern_s, TestNode._a, TestNode._p)
        roots = [self._expand(Node.root(TestNode._pi, TestNode._tve)),
                 Node.root(TestNode._pi, TestNode._tve)]
        o = unittest.mock.Mock(Observation)
        simulations = 1
        visitor = unittest.mock.Mock(
                NodeVisitor,
                **{'pre_visit_internal.return_value': 0, 
                   'post_visit_internal.return_value': 0, 
                   'visit_leaf.return_value': 0})
        call_counts_expected = [simulations, simulations + 1]
        call_counts_actual = []
        
        with self.assertRaises(NodeException) as cm:
            nonroot.mcts(o, simulations, visitor)
        
        for root in roots:
            root.mcts(o, simulations, visitor)
            call_counts_actual.append(visitor.visit_leaf.call_count)
            visitor.reset_mock()
        
        self.assertIs(cm.exception.get_exc_code(), NodeExceptionCode.NONROOT)  

        for root, call_count_actual, call_count_expected \
                in zip(roots, call_counts_actual, call_counts_expected):
            self.assertEqual(call_count_actual, call_count_expected)

    def test_update_stats(self) -> None:
        """Test `Node.test_update_stats()`.
        
        This method checks 4 cases:
        - An undetermined node.
        - An expanded root node.
        - An expanded non-root node.
        - A terminal node.
        
        Test case 1  
        **Given** an undetermined node  
        **When** the node update its statistics  
        **Then** the `NodeException` of the code `NodeException.UNDETERMINED`  
        should be raised.
        
        Test case 2  
        **Given** an expanded root node  
        **When** the node update its statistics  
        **Then** only the visit count of the node should be increased by 1.
        
        Test case 3  
        **Given** an expanded non-root node and the state value of the  
        next state  
        **When** the node update its statistics  
        **Then** the visit count of the node should be increased by 1 and the  
        discount factor * next state value should be averaged into the  
        action-value of the node.
        
        Test case 4  
        **Given** a terminal node and the state value of the next state
        **When** the node update its statistics  
        **Then** the visit count of the node should be increased by 1 and the  
        discount factor * next state value should be averaged into the  
        action-value of the node.
        """
        invalid_node = Node(TestNode._intern_s, TestNode._a, TestNode._p)
        valid_nodes = [
                self._expand(Node.root(TestNode._pi, TestNode._tve)),
                self._expand(
                        Node(TestNode._intern_s, TestNode._a, TestNode._p)),
                self._expand(
                        Node(TestNode._intern_s, TestNode._a, TestNode._p), 
                        True)]
        v = 0.5
        N_expected = [1, 1, 1]
        Q_expected = [0.5, 0.85, 0.85]
        Q_actual = []
        
        with self.assertRaises(NodeException) as cm:
            invalid_node.update_stats(v)
        
        for valid_node in valid_nodes:
            Q_actual.append(valid_node.update_stats(v))
        
        self.assertIs(cm.exception.get_exc_code(), 
                      NodeExceptionCode.UNDETERMINED)
        
        for valid_node, q_actual, q_expected, n_expected in \
                zip(valid_nodes, Q_actual, Q_expected, N_expected):
            self.assertEqual(valid_node.get_n(), n_expected)
            self.assertAlmostEqual(q_actual, q_expected)
    
    def test_obtain_stats_children(self) -> None:
        """Test `Node.obtain_stats_children()`.
        
        This method checks 3 cases:
        - A terminal node.
        - An undetermined node.
        - An expanded node.
         
        Test case 1  
        **Given** a terminal node  
        **When** the node obtains statistics from its children  
        **Then** `NodeException` of the code `NodeExceptionCode.UNEXPANDED`  
        should be raised.
        
        Test case 2  
        **Given** an undetermined node  
        **When** the node obtains statistics from its children  
        **Then** `NodeException` of the code `NodeExceptionCode.UNEXPANDED`  
        should be raised.
        
        Test case 3  
        **Given** an expanded node
        **When** the node obtains statistics from its children  
        **Then** prior probabilities, action-values, and visit counts of all  
        the children should be returned.
        """
        invalid_nodes = [
                self._expand(
                        Node(TestNode._intern_s, TestNode._a, TestNode._p),
                        True),
                Node(TestNode._intern_s, TestNode._a, TestNode._p)]
        valid_node  = self._expand(
                Node(TestNode._intern_s, TestNode._a, TestNode._p))
        stats_expected = {'P': [1.0], 'Q': [0], 'N': [0]}
        exc_codes_actual = []
        
        for invalid_node in invalid_nodes:
            with self.assertRaises(NodeException) as cm:
                invalid_node.obtain_stats_children() 
            
            exc_codes_actual.append(cm.exception.get_exc_code())
        
        stats_actual = valid_node.obtain_stats_children()

        for exc_code_actual in exc_codes_actual:
            self.assertEqual(exc_code_actual, NodeExceptionCode.UNEXPANDED)
        
        self.assertDictEqual(stats_actual, stats_expected)
        
    def test_get_child(self) -> None:
        """Test `Node.test_get_child()`.
        
        This method checks 4 cases:
        - A terminal node.
        - An undetermined node.
        - An expanded node with an index out of range.
        - An expanded node with an index in range.
        
        Test case 1  
        **Given** a terminal node  
        **When** the node get a child  
        **Then** `NodeException` of the code `NodeException.UNEXPANDED`  
        should be raised.
        
        Test case 2  
        **Given** an undetermined node  
        **When** the node get a child  
        **Then** `NodeException` of the code `NodeException.UNEXPANDED`  
        should be raised.
        
        Test case 3  
        **Given** an expanded node with an index out of range  
        **When** the node get the child  
        **Then** `IndexError` should be raised.
        
        Test case 4  
        **Given** an expanded node with an index in range  
        **When** the node get the child  
        **Then** the child of the index should be returned.
        """
        invalid_nodes = [
                self._expand(
                        Node(TestNode._intern_s, TestNode._a, TestNode._p), 
                        True),
                Node(TestNode._intern_s, TestNode._a, TestNode._p)]
        valid_node = self._expand(
                Node(TestNode._intern_s, TestNode._a, TestNode._p))
        idx_out_of_range = 1
        idx_in_range = 0
        exc_codes_actual = []
        
        for invalid_node in invalid_nodes:
            with self.assertRaises(NodeException) as cm:
                invalid_node.get_child(idx_in_range)

            exc_codes_actual.append(cm.exception.get_exc_code())
        
        with self.assertRaises(IndexError) as cm:
            valid_node.get_child(idx_out_of_range)
        
        child = valid_node.get_child(idx_in_range)
        
        for exc_code_actual in exc_codes_actual:
            self.assertIs(exc_code_actual, NodeExceptionCode.UNEXPANDED)
        
        self.assertEqual(str(cm.exception), 
                         'can only get 0th to 0th (not "1") child')
        self.assertIs(child, TestNode._child)
        
    def _expand(self, node: Node, make_terminal: bool=False) -> Node:
        """Expand given node.

        Args:
            node (Node): The node.
            make_terminal (bool, optional): Flag that indicates whether  
                expanded node becomes terminal or not. Defaults to `False`.
            
        Returns:
            Node: The expanded node.
        """
        if not node.is_root():
            r = unittest.mock.Mock(Reward, 
                                   **{'get_discount_factor.return_value': 0.9})
            r.__add__ = mock_r_add
            node.set_r(r)
            
        s = unittest.mock.Mock(State, 
                               **{'is_terminal.return_value': make_terminal})
        node.set_s(s) 
        
        if not make_terminal:
            node.add_child(TestNode._child)

        return node


def mock_r_add(self, other):
    r = 0.4
    
    return r + other
