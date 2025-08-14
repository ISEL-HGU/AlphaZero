"""Utility functions for tree search."""

import tensorflow as tf


def puct(P: tf.Tensor, Q: tf.Tensor, N_child: tf.Tensor, n_parent: int) \
        -> tf.Tensor:
    """Calculate upper confidence bound for trees with predictor.

    Note:
        - This function implements upper confidence bound for trees with
          predictor from "Mastering Atari, Go, chess and shogi by planning with
          a learned model" `(Schrittwieser et al., 2020)`_. 
        
    .. _(Schrittwieser et al., 2020): https://www.nature.com/articles/s41586-020-03051-4
    
    The formula for caculating the action number is as following:
    .. math::
        a=\\underset{a}{\\mathrm{argmax}}\\Bigg[Q(s,a)+P(s,a)\\frac{ \\
        \\sqrt{\\sum_{b}N(s,b)}}{1+N(s,a)}\\Bigg(c_1+\\log(\\frac{ \\
        \\sum_{b}N(s,b)+c_2+1}{c_2}))],
    
    where :math:`\\sum_{b}N(s,b)` denotes visit count of the parent node.  
    :math:`c_1=1.25` and :math:`c_2=19652` is used. 
    
    Args:
        P (tf.Tensor): Prior probabilities.
        Q (tf.Tensor): State-action values.
        N_child (tf.Tensor): Visit counts of children.
        n_parent (int): Visit count of parent.
    
    Returns:
        tf.Tensor: PUCT values.
    """
    C1 = 1.25
    C2 = 19652
    
    return Q + P * tf.math.sqrt(n_parent) \
                 / (1 + N_child) \
                 * (C1 + tf.math.log((N_child + C2 + 1) / C2))
    