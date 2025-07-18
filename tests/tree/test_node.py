"""Testcase for Node class."""

import unittest.mock

from alphazero.algorithms.policy_improvements.policy_improvement \
        import PolicyImprovement
from alphazero.algorithms.value_estimation.tree.tree_value_estimation \
        import TreeValueEstimation
from alphazero.exceptions.node_exception import NodeException, NodeExceptionCode
from alphazero.mdp.action import Action
from alphazero.mdp.reward import Reward
from alphazero.mdp.state import Observation, State
from alphazero.tree.node import Node
from alphazero.tree.visitor import NodeVisitor


class TestNode(unittest.TestCase):
    """Class that tests functionalities of `Node`.

    Attributes:
        _p (float): Arbitrary prior probability.
        _pi (unittest.mock.Mock): Mock policy improvement algorithm.
        _tve (unittest.mock.Mock): Mock tree value estimation algoritm.
        _discount_factor (float): Arbitrary discount factor.
    """
    
    @classmethod
    def setUpClass(cls) -> None:
        cls._intern_s = unittest.mock.Mock(State)
        cls._a = unittest.mock.Mock(Action)
        cls._p = 0.7
        cls._pi = unittest.mock.Mock(PolicyImprovement)
        cls._tve = unittest.mock.Mock(TreeValueEstimation)
        cls._discount_factor = 0.9
        
    def test_mcts(self) -> None:
        """Test `Node.mcts()`.
        
        This method checks 3 cases:
        - A non root node.
        - An expanded root node. 
        - An unexpanded root node.
        
        Test case 1  
        **Given** a non root node  
        **When** the node conducts mcts  
        **Then** `NodeExcpetion` should be raised.  
        
        Test case 2  
        **Given** an expanded root node and the number of simulations  
        **When** the node conducts mcts  
        **Then** the observation should be set to the node and the tree  
        traversing should be done for the number of simulations times.
        
        Test case 3  
        **Given** an unexpanded root node and the number of simulations  
        **When** the node conducts mcts  
        **Then** the observation should be set to the node and the tree  
        traversing should be done for the number of simulations+1 times. 
        """
        non_root = Node(self._intern_s, self._a, self._p, self._discount_factor)
        roots = [self._expand(
                    Node.root(self._pi, self._tve, self._discount_factor)),
                 Node.root(self._pi, self._tve, self._discount_factor)]
        o = unittest.mock.Mock(Observation)
        simulations = 1
        visitor = unittest.mock.Mock(NodeVisitor)
        visitor.configure_mock(
                **{'pre_visit_internal.return_value': 0, 
                    'post_visit_internal.return_value': 0, 
                    'visit_leaf.return_value': 0})
        call_counts_expected = [simulations, simulations + 1]
        call_counts_actual = []
        
        with self.assertRaises(NodeException) as cm:
            non_root.mcts(o, simulations, visitor)
        
        for root in roots:
            root.mcts(o, simulations, visitor)
            call_counts_actual.append(visitor.visit_leaf.call_count)
            visitor.reset_mock()
            
        self.assertEqual(cm.exception._exc_code, NodeExceptionCode.NON_ROOT)

        for root, call_count_actual, call_count_expected \
                in zip(roots, call_counts_actual, call_counts_expected):
            self.assertEqual(root._o, o)
            self.assertEqual(call_count_actual, call_count_expected)
            
    def test_add_child(self) -> None:
        """Test `Node.add_child()`.
        
        This method checks 2 cases:
        - A node that has no children.
        - A node that has at least one child.
        
        Test case 1  
        **Given** a node that has no children  
        **When** the node adds a child  
        **Then** the list with the added child should be set to the node.
        
        Test case 2  
        **Given** a node that has at least one child  
        **When** the node adds a child  
        **Then** the child should be added to the children list of the node.
        """
        nodes = [Node(self._intern_s, self._a, self._p, self._discount_factor),
                 self._expand(
                        Node(self._intern_s, self._a, 
                             self._p, self._discount_factor))]
        child = Node(self._intern_s, self._a, self._p, self._discount_factor)
        
        for node in nodes:
            node.add_child(child)
        
        for node in nodes:
            self.assertIs(node._children[-1], child)

    def test_update_stats(self) -> None:
        """Test `Node.test_update_stats()`.
        
        This method checks 3 cases:
        - An unexpanded node.
        - An expanded root node.
        - An expanded non root node.
        
        Test case 1  
        **Given** an unexpanded node  
        **When** the node update its statistics  
        **Then** the `NodeException` should be raised.
        
        Test case 2  
        **Given** an expanded root node  
        **When** the node update its statistics  
        **Then** only the visit count of the node should be increased by 1.
        
        Test case 3  
        **Given** an expanded non root node and the state value of the  
        previous state  
        **When** the node update its statistics  
        **Then** the visit count of the node should be increased by 1 and the  
        discount factor * previous state value should be averaged into the  
        action-value of the node.
        """
        unexpanded = Node(self._intern_s, self._a, 
                          self._p, self._discount_factor)
        expandeds = [self._expand(
                        Node.root(self._pi, self._tve, self._discount_factor)),
                     self._expand(
                        Node(self._intern_s, self._a, 
                             self._p, self._discount_factor))]
        v = 1.0
        q_n_expected = [(0, 1), (1.0, 1)]
        
        with self.assertRaises(NodeException) as cm:
            unexpanded.update_stats(v)
        
        for expanded in expandeds:
            expanded.update_stats(v)
        
        self.assertTrue(cm.exception._exc_code, NodeExceptionCode.UNEXPANDED)
        
        for expanded, (q_expected, n_expected) in zip(expandeds, q_n_expected):
            self.assertEqual(expanded._q, q_expected)
            self.assertEqual(expanded._n, n_expected)
    
    def test_obtain_stats_children(self) -> None:
        """Test `Node.obtain_stats_children()`.
        
        This method checks 3 cases:
        - An unexpanded node.
        - A (expanded) terminal node.
        - An expanded non terminal node.
        
        Test case 1  
        **Given** an unexpanded node  
        **When** the node obtains statistics from its children  
        **Then** `NodeException` should be raised.
        
        Test case 2  
        **Given** a terminal node  
        **When** the node obtains statistics from its children  
        **Then** `NodeException` should be raised.
        
        Test case 3  
        **Given** an expanded non terminal node  
        **When** the node obtains statistics from its children  
        **Then** prior probabilities, action-values, and visit counts of all  
        the children should be returned.
        """
        leafs = [Node(self._intern_s, self._a, self._p, self._discount_factor),
                 self._expand(
                        Node(self._intern_s, self._a, 
                             self._p, self._discount_factor), 
                        True)]
        expanded_non_terminal = self._expand(
                Node(self._intern_s, self._a, self._p, self._discount_factor))
        exc_codes_expected = [NodeExceptionCode.UNEXPANDED, 
                              NodeExceptionCode.TERMINAL]
        stats_expected = {'P': [self._p], 'Q': [0], 'N': [0]}
        exc_codes_actual = []
        
        for leaf in leafs:
            with self.assertRaises(NodeException) as cm:
                leaf.obtain_stats_children() 
            
            exc_codes_actual.append(cm.exception._exc_code)
        
        stats_actual = expanded_non_terminal.obtain_stats_children()
        
        for exc_code_actual, exc_code_expected \
                in zip(exc_codes_actual, exc_codes_expected):
            self.assertEqual(exc_code_actual, exc_code_expected)
        
        self.assertDictEqual(stats_actual, stats_expected)
        
    def test_get_child(self) -> None:
        """Test `Node.test_get_child()`.
        
        This method checks 4 cases:
        - An unexpanded node.
        - A (expanded) terminal node.
        - An expanded non terminal node with an index out of range.
        - An expanded non terminal node with an index in range.
        
        Test case 1  
        **Given** an unexpanded node  
        **When** the node get the child  
        **Then** `NodeException` should be raised.
        
        Test case 2  
        **Given** a terminal node  
        **When** the node get the child  
        **Then** `NodeException` should be raised.
        
        Test case 3  
        **Given** an expanded non terminal and an index out of range  
        **When** the node get the child  
        **Then** `IndexError` should be raised.
        
        Test case 4  
        **Given** an expanded non terminal node and an index in range  
        **When** the node get the child  
        **Then** the child of the index should be returned.
        """
        leaves = [Node(self._intern_s, self._a, self._p, self._discount_factor),
                  self._expand(
                        Node(self._intern_s, self._a, 
                             self._p, self._discount_factor), 
                        True)]
        expanded_non_terminal = self._expand(
                Node(self._intern_s, self._a, self._p, self._discount_factor))
        idx_out_of_range = -1
        idx_in_range = 0
        exc_codes_expected = [NodeExceptionCode.UNEXPANDED, 
                              NodeExceptionCode.TERMINAL]
        exc_codes_actual = []
        
        for leaf in leaves:
            with self.assertRaises(NodeException) as cm:
                leaf.get_child(idx_in_range)

            exc_codes_actual.append(cm.exception._exc_code)
        
        with self.assertRaises(IndexError) as cm:
            expanded_non_terminal.get_child(idx_out_of_range)
        
        child = expanded_non_terminal.get_child(idx_in_range)
        
        for exc_code_actual, exc_code_expected \
                in zip(exc_codes_actual, exc_codes_expected):
            self.assertEqual(exc_code_actual, exc_code_expected)
            
        self.assertEqual(cm.exception.args[0], 
                         'can only get 0th to 0th child (not "-1").') 
        self.assertIs(child, expanded_non_terminal._children[0])
        
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
            r = unittest.mock.Mock(Reward)
            r.__add__ = mock_r_add
            node._r = r
            
        s = unittest.mock.Mock(State)
        s.configure_mock(**{'is_terminal.return_value': make_terminal})
        node._s = s 
        
        if not make_terminal:
            node._children = [Node(self._intern_s, self._a, 
                                   self._p, self._discount_factor)]

        return node


def mock_r_add(self, other):
    return 0.1 + other
