"""
Markov Decision Processes. (Chapter 17)
First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid. We also represent a policy
as a dictionary of {state: action} pairs, and a Utility function as a
dictionary of {state: number} pairs. We then define the value_iteration
and policy_iteration algorithms.
"""
import random
import numpy as np
from mdp import MDP

class POMDP(MDP):
    """A Partially Observable Markov Decision Process, defined by
    a transition model P(s'|s,a), actions A(s), a reward function R(s),
    and a sensor model P(e|s). We also keep track of a gamma value,
    for use by algorithms. The transition and the sensor models
    are defined as matrices. We also keep track of the possible states
    and actions for each state. [Page 659]."""

    def __init__(self, actlist, init = None, terminals = None, transitions=None, evidences=None, rewards=None, states=None, gamma=0.95):
        """Initialize variables of the pomdp"""

        if not (0 < gamma <= 1):
            raise ValueError('A pomdp.pomdpy must have 0 < gamma <= 1')

        self.states = states
        self.actlist = actlist

        # transition model cannot be undefined
        self.transitions = transitions or {}
        if not self.transitions:
            print('Warning: Transition model is undefined')

        # sensor model cannot be undefined
        self.evidences = evidences or {}
        if not self.evidences:
            print('Warning: Sensor model is undefined')

        self.gamma = gamma
        self.rewards = rewards
        self.terminals = terminals
        self.current_state = init if init != None else random.choice(states - set(terminals))
        print("State: "+ str(self.current_state))

    def get_evidence(self, state):
        prop = random.random()
        evidence_value = 0.0
        i = 0
        self.evidences[state].sort(reverse=True)
        while True:
            evidence_value += self.evidences[state][i][0]
            if prop <= evidence_value:
                break
            i += 1
        return self.evidences[state][i][1]

    def current_state_is_terminal(self):
        return self.current_state in self.terminals

    def act(self, action):
        prop = random.random()
        action_value = 0.0
        i = 0
        while True:
            action_value += self.transitions[self.current_state][action][i][0]
            if prop <= action_value:
                break
            i+=1
        self.current_state = self.transitions[self.current_state][action][i][1]
        print("State: "+ str(self.current_state))
        return self.current_state in self.terminals, self.rewards[self.current_state], self.get_evidence(self.current_state)

    def remove_dominated_plans(self, input_values):
        """
        Remove dominated plans.
        This method finds all the lines contributing to the
        upper surface and removes those which don't.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = [values[0]]
        y1_max = max(val[1] for val in values)
        tgt = values[0]
        prev_b = 0
        prev_ix = 0
        while tgt[1] != y1_max:
            min_b = 1
            min_ix = 0
            for i in range(prev_ix + 1, len(values)):
                if values[i][0] - tgt[0] + tgt[1] - values[i][1] != 0:
                    trans_b = (values[i][0] - tgt[0]) / (values[i][0] - tgt[0] + tgt[1] - values[i][1])
                    if 0 <= trans_b <= 1 and trans_b > prev_b and trans_b < min_b:
                        min_b = trans_b
                        min_ix = i
            prev_b = min_b
            prev_ix = min_ix
            tgt = values[min_ix]
            best.append(tgt)

        return self.generate_mapping(best, input_values)

    def remove_dominated_plans_fast(self, input_values):
        """
        Remove dominated plans using approximations.
        Resamples the upper boundary at intervals of 100 and
        finds the maximum values at these points.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = []
        sr = 100
        for i in range(sr + 1):
            x = i / float(sr)
            maximum = (values[0][1] - values[0][0]) * x + values[0][0]
            tgt = values[0]
            for value in values:
                val = (value[1] - value[0]) * x + value[0]
                if val > maximum:
                    maximum = val
                    tgt = value

            if all(any(tgt != v) for v in best):
                best.append(np.array(tgt))

        return self.generate_mapping(best, input_values)

    def generate_mapping(self, best, input_values):
        """Generate mappings after removing dominated plans"""

        mapping = defaultdict(list)
        for value in best:
            for action in input_values:
                if any(all(value == v) for v in input_values[action]):
                    mapping[action].append(value)

        return mapping

    def max_difference(self, U1, U2):
        """Find maximum difference between two utility mappings"""

        for k, v in U1.items():
            sum1 = 0
            for element in U1[k]:
                sum1 += sum(element)
            sum2 = 0
            for element in U2[k]:
                sum2 += sum(element)
        return abs(sum1 - sum2)





