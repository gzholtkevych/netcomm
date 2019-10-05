import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import exit
from utils import *


nvars = 2  # number of choice variants
# ----------------------------------------------------------
# community specification
# ----------------------------------------------------------
# specify community net graph
@decorator_for_specifying_community_graph
def define_community():
    # The code of this function depends on an experiment.
    # The function realises creating for the experiment
    #   the concrete graph of the network community.
    # The created graph should be returned.
    return nx.complete_graph(200)
net = define_community(nvars)
# ----------------------------------------------------------
# assign parameter values of community actors
@decorator_for_assigning_actors_parameter_values
def assign_actors_parameter_values(
    net  # the created community graph
):
    # The code of this function depends on an experiment.
    # The function realises assigning for the experiment
    #   community actors parameter values.
    # Default values: rho = 1.0 and choice = DISCLAIMER.
    # The function returns None.
    for n in net:
        net.nodes[n]['rho'] = 20
    net.nodes[0]['choice'] = 0
assign_actors_parameter_values(net)
# ----------------------------------------------------------
# assign parameter values of community channels
@decorator_for_assigning_channels_parameter_values
def assign_channels_parameter_values(
    net  # the created community graph
):
    # The code of this function depends on an experiment.
    # The function realises assigning for the experiment
    #   community channels parameter values.
    # Default values: a = 1.0 and D = [[0.5, 0.5],
    #                                  [0.5, 0.5]].
    # The function returns None.
    for channel in net.edges:
        alice, _ = actor_roles(channel)
        if alice == 0:
            net.edges[channel]['D'] = \
                define_dialogue_matrix(1.0,
                    rg.uniform(low=0.2, high=0.6)
                )
        else:
            net.edges[channel]['D'] = \
                define_dialogue_matrix(
                    rg.uniform(low=0.2, high=0.6),
                    rg.uniform(low=0.2, high=0.6)
                )
assign_channels_parameter_values(net)

# # auxiliary parameters initialisation
# for n in net.nodes:
#     net.nodes[n]['result_list'] = []
# # ----------------------------------------------------------


# #
# def simulate_dialog(alice, bob) :
#     global net
# # get dialogue matrix of the current dialogue and
# # the preference densities of its participants
#     D = net.edges[alice, bob]['D']
#     wA = net.nodes[alice]['w']
#     wB = net.nodes[bob]['w']
#     wA_result = np.zeros(net.nvars)
#     wB_result = np.zeros(net.nvars)
#     for v in range(net.nvars):
#         wA_result[v] = D[0, 0] * wA[v] + D[0, 1] * wB[v]
#         wB_result[v] = D[1, 0] * wA[v] + D[1, 1] * wB[v]
#     return wA_result, wB_result


# #
# def simulate_session():
#     global net
# # clean auxiliary information
#     for channel in net.edges:
#         for ic in channel:
#             net.nodes[ic]['result_list'][:] = []
# # simulate session dialogues
#     for channel in net.edges:
#         if not Bernoulli_trial(net.edges[channel]['a']):
#         # channel is not active
#             continue
#         else:  # channel active
#             pass
#     # determine actors participating as Alice and Bob in the
#     # current dialogue
#         alice, bob = min(channel), max(channel)
#     # ------------------------------------------------------
#         wA, wB = simulate_dialog(alice, bob)
#         net.nodes[alice]['result_list'].append(wA)
#         net.nodes[bob]['result_list'].append(wB)
# # compute the previous session result for each community actor
#     for n in net:
#         if net.nodes[n]['result_list']:
#         # actor 'n' participates at least in one dealogue
#             ndialogues = len(net.nodes[n]['result_list'])
#             w = np.zeros(net.nvars)
#             for wc in net.nodes[n]['result_list']:
#                 np.add(w, wc, w)
#             np.multiply(w, 1.0 / ndialogues,
#                 net.nodes[n]['w'])


# #
# def observation():
# # polling simulation
#     for n in net:
#         hn = h(net.nodes[n]['w'])
#         if Bernoulli_trial(
#                 np.power(hn, net.nodes[n]['rho'])):
#         # actor 'n' disclaims a choice
#             net.nodes[n]['choice'] = DISCLAIMER
#         else:
#         # actor 'n' chooses
#             net.nodes[n]['choice'] = np.random.choice(
#                 net.nvars, p=net.nodes[n]['w'])
# # compute average preference density
#     W = np.zeros(net.nvars)
#     for n in net:
#         np.add(W, net.nodes[n]['w'], W)
#     np.multiply(W, 1.0 / net.number_of_nodes(), W)
# # compute polling result
#     DP = len([1 for n in net
#             if net.nodes[n]['choice'] == DISCLAIMER])
#     if DP == net.number_of_nodes():
#     # all community actors disclaimed a choice 
#         return W, 1.0, uncertainty(net.nvars)
#     NP = net.number_of_nodes() - DP
#     WP = net.nvars * [None]
#     for v in range(net.nvars):
#         WP[v] = len([1 for n in net
#             if net.nodes[n]['choice'] == v])
#         WP[v] /= NP
#     DP /= net.number_of_nodes()
#     return W, DP, WP


# # ----------------------------------------------------------
# # experiment specification
# # ----------------------------------------------------------

# # specify initial prefernce densities of community actors
# for n in net:
#     if n == 0:
#         net.nodes[n]['w'] = np.array([1.0, 0.0], float)
#     elif n == 1:
#         net.nodes[n]['w'] = np.array([0.0, 1.0], float)
#     else:
#         net.nodes[n]['w'] = uncertainty(net.nvars)

# niter = 1000  # define number of iterations

# # set up the experiment

# protocol = [observation()]
# for istep in range(niter):
#     simulate_session()
#     protocol.append(observation())


# # ----------------------------------------------------------
# # store the experiment outcomes
# # ----------------------------------------------------------
# out_file = open("protocol.dat", "w")
# # out_file.write(str(net.nvars) + "\n")
# for item in protocol:
#     for val in item[0]:
#         out_file.write(str(val) + " ")
#     out_file.write(str(item[1]))
#     for val in item[2]:
#         out_file.write(" " + str(val))
#     out_file.write("\n")
# out_file.close()
