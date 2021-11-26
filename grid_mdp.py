from utils import vector_add, orientations, turn_right, turn_left, EAST, NORTH, WEST, SOUTH
from mdp import MDP, best_policy, policy_iteration, value_iteration

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


# ______________________________________________________________________________


""" [Figure 17.1]
A 4x3 grid environment that presents the agent with a sequential decision problem.
"""
r = -0.04
sequential_decision_environment = GridMDP([[r, r, r, +1],
                                           [r, None, r, -1],
                                           [r, r, r, r]],
                                          terminals=[(3, 2), (3, 1)])

sequential_decision_environment_big = GridMDP([[r, r, r, r,r,r,+1,r,r],
                                           [r, None, r, None, None,-1, None,r,r],
                                           [r, r, r, r,r,r,r,r,r]],
                                          terminals=[(6, 2), (7, 1)])
# ______________________________________________________________________________

pi = best_policy(sequential_decision_environment_big, value_iteration(sequential_decision_environment_big, .01))
sequential_decision_environment_big.to_arrows(pi)
from utils import print_table
print_table(sequential_decision_environment_big.to_arrows(pi))
print("\n")
pi = policy_iteration(sequential_decision_environment)
sequential_decision_environment.to_arrows(pi)
print_table(sequential_decision_environment.to_arrows(pi))