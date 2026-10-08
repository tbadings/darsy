import numpy as np

class IMDP:
    """
    Class to construct the IMDP abstraction.
    """

    def __init__(self, partition, states, x0, goal_regions, critical_regions, P_full, S_id, A_id, P_absorbing,
                 coarse_partition, Q_id, P_coarse):
        '''
        Generate the IMDP abstraction

        :param partition:
        :param states:
        :param x0:
        :param goal_regions:
        :param critical_regions:
        :param P_full:
        :param S_id:
        :param A_id:
        :param P_absorbing:
        :param coarse_partition: CoarsePartition grouping fine cells into blocks.
        :param Q_id: Per state and action, successor IDs of the block window ([A, Mc]).
        :param P_coarse: Per state and action, block probability intervals ([A, Mc, 2]).
        '''
        
        self.states = states

        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.P_full = P_full
        self.S_id = S_id
        self.A_id = A_id
        self.P_absorbing = P_absorbing

        # Coarse layer
        self.coarse_partition = coarse_partition
        self.Q_id = Q_id
        self.P_coarse = P_coarse

        # Fine/coarse maps
        self.parents = coarse_partition.parents
        self.children = coarse_partition.children

        # Define initial state
        self.s_init = partition.x2state(x0)[0]

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1