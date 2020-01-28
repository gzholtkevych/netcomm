import numpy as np
import numpy.random as rnd
import networkx as nx


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


def decorate_specifying_community_graph(
    specifying_function
):
    def wrapped_function(nvars):
        # create net
        net = specifying_function()
        # associte 'nvars' with 'net'
        setattr(net, 'nvars', nvars)
        setattr(net, 'W', None)
        setattr(net, 'DP', None)
        setattr(net, 'WP', None)
        return net
    return wrapped_function


def decorate_assigning_actors_parameter_values(
    assigning_function
):
    def wrapped_function(net):
        for n in net:  # default assigning
            net.nodes[n]['choice'] = DISCLAIMER
            net.nodes[n]['rho'] = 1.0
            # the list for storing auxiliary data
            #   of the node
            net.nodes[n]['aux_list'] = []
        assigning_function(net)
    return wrapped_function


def decorate_assigning_channels_parameter_values(
    assigning_function
):
    def wrapped_function(net):
        for channel in net.edges:
            net.edges[channel]['a'] = 1.0
            net.edges[channel]['D'] = \
                define_dialogue_matrix(0.5, 0.5)
        assigning_function(net)
    return wrapped_function


def simulate_dialog(
    net,     # a network community
    channel  # a communication channel
):  # returns the preference densities of the current
    # dialogue participants after the dialogue
    alice, bob = actor_roles(channel)
    # get dialogue matrix of the current dialogue and
    # the preference densities of its participants
    # before the dialogue
    D = net.edges[channel]['D']
    wAlice = net.nodes[alice]['w']
    wBob = net.nodes[bob]['w']
    # initialisation the preference densities of the current
    # dialogue participants after the dialogue
    wAlice_after, wBob_after = \
        np.zeros(net.nvars), np.zeros(net.nvars)
    for v in range(net.nvars):
        wAlice_after[v], wBob_after[v] = \
            D[0, 0] * wAlice[v] + D[0, 1] * wBob[v], \
            D[1, 0] * wAlice[v] + D[1, 1] * wBob[v]
    return wAlice_after, wBob_after


def clear_aux_list(net):
    for n in net:
        net.nodes[n]['aux_list'][:] = []


def simulate_session(
    net  # a network community
):  # returns None
    clear_aux_list(net)
    for channel in net.edges:  # simulate session dialogues
        if Bernoulli_trial(net.edges[channel]['a']):
        # channel is active
            pass
        else:  # channel is not active
            continue
        alice, bob = actor_roles(channel)
        wAlice, wBob = simulate_dialog(net, channel)
        net.nodes[alice]['aux_list'].append(wAlice)
        net.nodes[bob]['aux_list'].append(wBob)
    for n in net:  # compute the session results
        if net.nodes[n]['aux_list']:
        # actor 'n' participates at least in one dealogue
            ndialogues = len(net.nodes[n]['aux_list'])
            w = np.zeros(net.nvars)
            for wc in net.nodes[n]['aux_list']:
                np.add(w, wc, w)
            np.multiply(w, 1.0 / ndialogues,
                net.nodes[n]['w'])


def observation(
    net,
    specific_observation_function
):  # returns community preference density,
    #         disclaimer probability, and
    #         estimated community preference density
    for n in net:  # polling simulation
        hn = h(net.nodes[n]['w'])
        if Bernoulli_trial(
                np.power(hn, net.nodes[n]['rho'])):
        # actor 'n' disclaims a choice
            net.nodes[n]['choice'] = DISCLAIMER
        else:
        # actor 'n' chooses
            net.nodes[n]['choice'] = \
                np.random.choice(
                    net.nvars, p=net.nodes[n]['w']
            )
    # compute community preference density
    net.W = np.zeros(net.nvars)
    for n in net:
        np.add(net.W, net.nodes[n]['w'], net.W)
    np.multiply(net.W, 1.0 / net.number_of_nodes(), net.W)
    # compute polling result
    DN = len([1 for n in net  # number of disclaiming nodes
                if net.nodes[n]['choice'] == DISCLAIMER])
    if DN == net.number_of_nodes():
    # all community actors disclaimed a choice 
        net.DP = 1.0
        net.WP = uncertainty(net.nvars)
    else:
        net.DP = DN / net.number_of_nodes()
        NP = net.number_of_nodes() - DN
        net.WP = net.nvars * [None]
        for v in range(net.nvars):
            net.WP[v] = len([1 for n in net
                               if net.nodes[n]['choice'] == v])
            net.WP[v] /= NP
    return specific_observation_function(net)


def decorate_specifying_initial_prefernce_densities(
    specifying_function
):
    def wrapped_function(net):
        for n in net:
            net.nodes[n]['w'] = uncertainty(net.nvars)
        specifying_function(net)
    return wrapped_function
