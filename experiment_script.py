import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import exit
from utils import *


# ----------------------------------------------------------
# community specification
# ----------------------------------------------------------
# specify community net graph
@decorate_specifying_community_graph
def define_community():  # returns the community graph
    # The code of this function depends on an experiment.
    # The function realises creating for the experiment
    #   the concrete graph of the network community.
    return nx.complete_graph(200)
# ----------------------------------------------------------
# assign parameter values of community actors
@decorate_assigning_actors_parameter_values
def assign_actors_parameter_values(
    net  # the community graph
):  # returns None
    # The code of this function depends on an experiment.
    # The function realises assigning for the experiment
    #   community actors parameter values.
    # Default values: rho = 1.0 and choice = DISCLAIMER.
    # The function returns None.
    for n in net:
        net.nodes[n]['rho'] = 20
    net.nodes[0]['choice'] = 0
# ----------------------------------------------------------
# assign parameter values of community channels
@decorate_assigning_channels_parameter_values
def assign_channels_parameter_values(
    net  # the community graph
):  # returns None
    # The code of this function depends on an experiment.
    # The function realises assigning for the experiment
    #   community channels parameter values.
    # Default values: a = 1.0 and D = [[0.5, 0.5],
    #                                  [0.5, 0.5]].
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
# ----------------------------------------------------------


# ----------------------------------------------------------
# initialize the netwoork community for studying
# ----------------------------------------------------------
nvars = 2                        # number of choice variants
net = define_community(nvars)
assign_actors_parameter_values(net)
assign_channels_parameter_values(net)
# ----------------------------------------------------------


# ----------------------------------------------------------
# experiment specification
# ----------------------------------------------------------

# specify initial prefernce densities of community actors
@decorate_specifying_initial_prefernce_densities
def specify_initial_preference_densities(
    net  # the network community under stufying
):  # returns None
    # The code of this function depends on an experiment.
    # The function realises specifying for the experiment
    #   community actors specific preference densities.
    # Default values: w = [1 / nvars, ..., 1 / nvars].
    # If this is what you need then uncomment the next line
    # pass
    # Otherwise insert your code below
    net.nodes[0]['w'] = np.array([1.0, 0.0], float)
    net.nodes[1]['w'] = np.array([0.0, 1.0], float)
# ----------------------------------------------------------


specify_initial_preference_densities(net)
niter = 1000                          # number of iterations

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
