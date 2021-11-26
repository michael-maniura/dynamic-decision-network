from grid_pomdp import GridPOMDP
import random

class DynamicDecisionNetwork:
    def __init__(self, max_depth):
        self.grid_pomdb = self.initialize_grid()
        self.belief_state = {
            (0,0) : 1/9,
            (0,1) : 1/9,
            (0,2) : 1/9,
            (1,0) : 1/9,
            (1,2) : 1/9,
            (2,0) : 1/9,
            (2,1) : 1/9,
            (2,2) : 1/9,
            (3,0) : 1/9,
            (3,1) : 0,
            (3,2) : 0
        }

        self.max_depth = max_depth
        self.possible_evidences = [0, 1, 2, 3]

    def initialize_grid(self):
        r = -0.04
        grid = GridPOMDP(
            [[r, r, r, +1],
            [r, None, r, -1],
            [r, r, r, r]],
            terminals=[(3, 2), (3, 1)], init=(0,0))
        return grid

    def solve_grid(self):
        self.get_maximum_utility_of_belief_state(self.belief_state, 0)

    def reached_terminal_state(self, belief_state):
        probability_sum = 0
        for terminal in self.grid_pomdb.terminals:
            probability_sum = probability_sum + belief_state[terminal]
        
        if probability_sum == 1:
            return True
        else:
            return False

    def get_maximum_utility_of_belief_state(self, belief_state, depth):
        if self.reached_terminal_state(belief_state):
            return self.get_belief_state_reward(belief_state)
        elif depth == self.max_depth:
            return self.get_belief_state_reward(belief_state)
        else:
            best_action, best_new_belief_state = self.get_best_action_and_new_belief_state_for_belief_state(belief_state)
            print("Best action: " + best_action)
            return self.get_maximum_utility_of_belief_state(best_new_belief_state, depth+1)
    
    def get_best_action_and_new_belief_state_for_belief_state(self, current_belief_state):
        best_new_belief_state = None
        best_new_belief_state_reward = 0
        best_action = None

        for action in random.sample(self.grid_pomdb.actlist, len(self.grid_pomdb.actlist)):
            for evidence in self.possible_evidences:
                new_possible_belief_state = self.get_new_belief_state(current_belief_state, action, self.grid_pomdb.evidences[evidence])
                belief_state_reward = self.get_belief_state_reward(new_possible_belief_state)
                
                if(belief_state_reward > best_new_belief_state_reward):
                    best_new_belief_state = new_possible_belief_state
                    best_new_belief_state_reward = belief_state_reward
                    best_action = action

        return (best_action, best_new_belief_state)

    def get_new_belief_state(self, current_belief_state, action, evidence):
        new_belief_state = {}
        for state in self.grid_pomdb.states:
            new_state_possibility = self.filter_algorithm(current_belief_state, state, evidence, action)
            new_belief_state[state] = new_state_possibility

    def filter_algorithm(self, current_belief_state, new_state, evidence, action):
        normalization_factor = self.get_normalization_factor(current_belief_state, new_state)
        probability_of_evidence_in_new_state = self.grid_pomdb.evidences[new_state][evidence][0]

        state_sum = 0
        for state in self.grid_pomdb.states:
            probability_to_reach_new_state_from_current_state_with_given_action = self.get_probability_to_reach_new_state_from_current_state_with_given_action(new_state, state, action)
            probability_of_state = current_belief_state[state]
            state_sum = state_sum + (probability_to_reach_new_state_from_current_state_with_given_action * probability_of_state)

        return normalization_factor * probability_of_evidence_in_new_state * state_sum

    def get_normalization_factor(self, current_belief_state, new_state):
        current_belief_state_probability_of_new_state = current_belief_state[new_state]
        
        sum_of_possible_follow_up_state_probabilities = 0
        for possible_follow_up_state in self.get_possible_follow_up_states(new_state):
            sum_of_possible_follow_up_state_probabilities = sum_of_possible_follow_up_state_probabilities + current_belief_state[possible_follow_up_state]

        if(sum_of_possible_follow_up_state_probabilities == 0):
            return 0
        else:
            return current_belief_state_probability_of_new_state / sum_of_possible_follow_up_state_probabilities
    
    def get_possible_follow_up_states(self, state):
        possible_follow_up_states = []
        
        for action in self.grid_pomdb.transitions[state]:
            for probability, possible_follow_up_state in self.grid_pomdb.transitions[state][action]:
                if(possible_follow_up_state not in possible_follow_up_states):
                    possible_follow_up_states.append(possible_follow_up_state)
        
        return possible_follow_up_states


    def get_probability_to_reach_new_state_from_current_state_with_given_action(
        self,
        new_state,
        current_state,
        action):

        for probability, possible_new_state in self.grid_pomdb.transitions[current_state][action]:
            if(possible_new_state == new_state):
                return probability
        return 0

    def get_belief_state_reward(self, belief_state):
        state_reward_sum = 0
        
        for state in belief_state:
            state_reward = self.grid_pomdb.rewards[state]
            weighted_state_reward = state_reward * belief_state[state]
            state_reward_sum = state_reward_sum + weighted_state_reward
        
        return state_reward_sum 

    # bellmann 
    def get_utility_of_state(self, state, visited_states):
        # inquire reward of state
        reward_current_state = self.grid_pomdb.rewards[state]
        
        best_follow_up_state_utility = 0
        # get best follow up state utility (recursion)
        for action in self.grid_pomdb.transitions[state]:
            action_state_transitions = self.grid_pomdb.transitions[state][action]
            for state_transition in action_state_transitions:
                possible_state_probability = state_transition[0]
                possible_state = state_transition[1]

                if(possible_state in visited_states):
                    continue
                else:
                    visited_states.append(possible_state)

                possible_state_utility = self.get_utility_of_state(possible_state, visited_states)
                weighted_possible_state_utility = possible_state_utility * possible_state_probability

                if(weighted_possible_state_utility > best_follow_up_state_utility):
                    best_follow_up_state_utility = weighted_possible_state_utility
                    
        # return reward of state + (result of above)
        return reward_current_state + best_follow_up_state_utility

ddn = DynamicDecisionNetwork(5)
ddn.solve_grid()