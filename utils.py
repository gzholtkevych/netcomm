import numpy as np
import numpy.random as rnd


DISCLAIMER = -1
rg = np.random.default_rng()


def actor_roles(
    channel  # graph edge
):
    return min(channel), max(channel)


def Bernoulli_trial(p):
    return rnd.choice([True, False], p=[p, 1- p])

# def readint(
#     prompt,    # prompt string
#     condition  # correctness condition
# ):
#     temp = input(prompt)
#     try:
#         temp = int(temp)
#     except:
#         raise ValueError("'{}' is not integer".format(temp))
#     if not condition(temp):
#         raise ValueError(
#             "choosen number '{}' is not valid".format(temp))
#     return temp

def uncertainty(m):
    return np.array(m * [1.0 / m], float)

def define_dialogue_matrix(p, q):
    assert isinstance(p, float), \
           "the 1st parameter is not float"
    assert isinstance(q, float), \
           "the 2nd parameter is not float"
    assert 0 <= p <= 1, \
           "the 1st parameter is out of the segment [0, 1]"
    assert 0 <= q <= 1, \
           "the 2nd parameter is out of the segment [0, 1]"
    return np.array([p, 1 - p, 1 - q, q], float).reshape(2, 2)

def h(w):  # the normalised entropy of distribution 'w'
    h = float(0)
    for p in w:
        if p :
            h -= p * np.log2(p)
    return h

def input_non_empty(prompt):
    while True:
        temp = input(prompt)
        if temp:
            return temp
        else:
            print("Error!", end=' ')


def decorator_for_specifying_community_graph(
    specifying_function
):
    def wrapped_function(nvars):
        # create net
        net = specifying_function()
        # associte 'nvars' with 'net'
        setattr(net, 'nvars', nvars)
        return net
    return wrapped_function


def decorator_for_assigning_actors_parameter_values(
    assigning_function
):
    def wrapped_function(net):
        for n in net:
            net.nodes[n]['choice'] = DISCLAIMER
            net.nodes[n]['rho'] = 1.0
        assigning_function(net)
    return wrapped_function


def decorator_for_assigning_channels_parameter_values(
    assigning_function
):
    def wrapped_function(net):
        for channel in net.edges:
            net.edges[channel]['a'] = 1.0
            net.edges[channel]['D'] = \
                define_dialogue_matrix(0.5, 0.5)
        assigning_function(net)
    return wrapped_function
