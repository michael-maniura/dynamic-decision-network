from pomdp import POMDP
from utils import vector_add, orientations, turn_right, turn_left, EAST, NORTH, WEST, SOUTH, Matrix
from collections import defaultdict

class GridPOMDP(POMDP):
    """Added perception to the GridMDP. The Agent does not know where he begins (only that its not a terminal state).
    He gets perceptions by a sensor that tells him the amount of walls but makes an error with
    probability = perception_failure.
     """

    def __init__(self, grid, terminals, init = None, perception_failure = .1, gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        rewards = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    rewards[(x, y)] = grid[y][x]
        self.states = states
        self.actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in self.actlist:
                transitions[s][a] = self.calculate_T(s, a)
        evidences ={}
        for s in states:
            evidences[s] = self.calculate_evidence(s, perception_failure)

        POMDP.__init__(self, actlist=self.actlist, init = init,
                     terminals = terminals, transitions=transitions, evidences = evidences,
                     rewards = rewards, states=states, gamma=gamma)

    def calculate_evidence(self, state, perception_failure):
        # tests movements in every direction (every action). For each failed move, there is a wall
        walls = self.get_walls_count(state)
        evidence = []
        prob = 0
        for w in range(4):
            prob = 0.0
            if walls == w:
                prob = 1.0 - perception_failure
            elif walls-1 == w or walls +1 == w:
               prob =  perception_failure/2
            evidence.append((prob, w))
        return evidence

    def get_walls_count(self, state):
        walls = 0
        for a in self.actlist:
            if self.simulate_go(state,a) == state:
                walls += 1
        return walls


    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.simulate_go(state, action)),
                    (0.1, self.simulate_go(state, turn_right(action))),
                    (0.1, self.simulate_go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def simulate_go(self, state, direction):
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


def pomdp_value_iteration(pomdp, epsilon=0.1):
    """Solving a pomdp.pomdpy by value iteration."""

    U = {'': [[0] * len(pomdp.states)]}
    count = 0
    while True:
        count += 1
        prev_U = U
        values = [val for action in U for val in U[action]]
        value_matxs = []
        for i in values:
            for j in values:
                value_matxs.append([i, j])

        U1 = defaultdict(list)
        for action in pomdp.actlist:
            for u in value_matxs:
                u1 = Matrix.matmul(Matrix.matmul(pomdp.transitions[int(action)],
                                                 Matrix.multiply(pomdp.evidences[int(action)], Matrix.transpose(u))),
                                   [[1], [1]])
                u1 = Matrix.add(Matrix.scalar_multiply(pomdp.gamma, Matrix.transpose(u1)), [pomdp.rewards[int(action)]])
                U1[action].append(u1[0])

        U = pomdp.remove_dominated_plans_fast(U1)
        # replace with U = pomdp.remove_dominated_plans(U1) for accurate calculations

        if count > 10:
            if pomdp.max_difference(U, prev_U) < epsilon * (1 - pomdp.gamma) / pomdp.gamma:
                return U

"""
r = -0.4
env = GridPOMDP([
                [-1,r,r,r,-1],
                 [None,None,1,None,None],],
              terminals=[(0, 1), (4, 1), (2,0)], init=(1,1))

r = -0.04
env = GridPOMDP([[r,r,+1],
               [r, r, None],
               [r, r,-1]],
              terminals=[(2, 2), (2, 0)], init=(0,1))

"""
r = -0.04
env = GridPOMDP([[r, r, r, +1],
               [r, None, r, -1],
               [r, r, r, r]],
              terminals=[(3, 2), (3, 1)], init=(0,0))