from grid_pomdp import GridPOMDP
import random
import time

class DynamicDecisionNetwork:
    def __init__(self, max_depth):
        """
        Initializer function of the class with
        a maximum depth parameter for the
        belief state forward-chaining
        """
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
        self.possible_evidence_indices = [0, 1, 2, 3]

    def initialize_grid(self):
        """
        Initialize the POMDB grid
        to solve
        """
        r = -0.04
        grid = GridPOMDP(
            [[r, r, r, +1],
            [r, None, r, -1],
            [r, r, r, r]],
            terminals=[(3, 2), (3, 1)], init=(0,0))
        return grid

    def solve_grid(self):
        """
        Solve the initialized grid
        """
        print("Starting grid solving...")
        while not self.reached_terminal_state(self.belief_state):
            best_action, new_belief_state, reward = self.get_maximum_utility_of_belief_state(self.belief_state, 0)
            self.perform_action_and_update_belief_state(best_action, new_belief_state)
        print("Solved the grid successfully!")

    def perform_action_and_update_belief_state(self, action, new_belief_state):
        """
        Print the best action of a step
        and the new, assigned belief state.
        """
        print("Performing best action: %s" % (action,))
        print("New belief state:\n%s" % (new_belief_state,))
        time.sleep(5)
        self.belief_state = new_belief_state

    def get_maximum_utility_of_belief_state(self, belief_state, depth):
        """
        Calculate the maximum utility
        of a given belief state.
        This function calls itself recursively
        with an incrementing depth parameter
        until the class instance specific
        maximum depth is reached  
        """
        if self.reached_terminal_state(belief_state):
            return (None, None, self.get_belief_state_reward(belief_state))
        elif depth == self.max_depth:
            return (None, None, self.get_belief_state_reward(belief_state))
        else:
            best_action, best_new_belief_state = self.get_best_action_and_new_belief_state_for_belief_state(belief_state)
            return (best_action, best_new_belief_state, self.get_maximum_utility_of_belief_state(best_new_belief_state, depth+1)[2])
    
    def reached_terminal_state(self, belief_state):
        """
        Check if a terminal belief state
        has been reached.
        """
        probability_sum = 0
        for terminal in self.grid_pomdb.terminals:
            probability_sum = probability_sum + belief_state[terminal]
        
        if probability_sum == 1:
            return True
        else:
            return False

    def get_best_action_and_new_belief_state_for_belief_state(self, belief_state):
        """
        Get the best action to execute
        and the resulting belief state
        from a given belief state
        """
        best_new_belief_state = None
        best_new_belief_state_reward = 0
        best_action = None

        action_belief_state_reward = []

        for action in random.sample(self.grid_pomdb.actlist, len(self.grid_pomdb.actlist)):
            for evidence_index in self.possible_evidence_indices:
                new_possible_belief_state = self.get_new_belief_state(belief_state, action, evidence_index)
                belief_state_reward = self.get_belief_state_reward(new_possible_belief_state)
                
                action_belief_state_reward.append((action, new_possible_belief_state, belief_state_reward))

                """
                if belief_state_reward > best_new_belief_state_reward:
                    best_new_belief_state = new_possible_belief_state
                    best_new_belief_state_reward = belief_state_reward
                    best_action = action
                """

        best_action, best_new_belief_state, best_new_belief_state_reward = max(action_belief_state_reward, key=lambda element:element[2])
        return (best_action, best_new_belief_state)

    def get_new_belief_state(self, belief_state, action, evidence_index):
        """
        Calculate a new belief state
        from a given belief state
        for the execution of a given action
        with given evidence index
        """
        new_belief_state = {}
        for state in self.grid_pomdb.states:
            new_state_possibility = self.get_probability_of_new_state_in_new_belief_state(belief_state, state, evidence_index, action)
            new_belief_state[state] = new_state_possibility
        return new_belief_state

    def get_probability_of_new_state_in_new_belief_state(self, belief_state, new_state, evidence_index, action):
        """
        The filter algorithm
        """
        normalization_factor = self.get_normalization_factor(belief_state, new_state)
        probability_of_evidence_in_new_state = self.grid_pomdb.evidences[new_state][evidence_index][0]

        state_sum = 0
        for state in self.grid_pomdb.states:
            probability_to_reach_new_state_from_current_state_with_given_action = self.get_probability_to_reach_new_state_from_current_state_with_given_action(new_state, state, action)
            probability_of_state = belief_state[state]
            state_sum = state_sum + (probability_to_reach_new_state_from_current_state_with_given_action * probability_of_state)

        return normalization_factor * probability_of_evidence_in_new_state * state_sum

    def get_normalization_factor(self, belief_state, new_state):
        """
        Get the normalization factor
        for a given belief state and
        a specific state for the filter
        algorithm
        """
        current_belief_state_probability_of_new_state = belief_state[new_state]
        
        sum_of_possible_follow_up_state_probabilities = 0
        for possible_follow_up_state in self.get_possible_follow_up_states(new_state):
            sum_of_possible_follow_up_state_probabilities = sum_of_possible_follow_up_state_probabilities + belief_state[possible_follow_up_state]

        if sum_of_possible_follow_up_state_probabilities == 0:
            return 0
        else:
            return current_belief_state_probability_of_new_state / sum_of_possible_follow_up_state_probabilities
    
    def get_possible_follow_up_states(self, state):
        """
        Get all possible states which can be
        reached from a given state
        """
        possible_follow_up_states = []
        
        for action in self.grid_pomdb.transitions[state]:
            for probability, possible_follow_up_state in self.grid_pomdb.transitions[state][action]:
                if possible_follow_up_state not in possible_follow_up_states:
                    possible_follow_up_states.append(possible_follow_up_state)
        
        return possible_follow_up_states


    def get_probability_to_reach_new_state_from_current_state_with_given_action(
        self,
        new_state,
        current_state,
        action):
        """
        Get the probability to reach
        a new given state from a given
        current state with a given action
        """
        for probability, possible_new_state in self.grid_pomdb.transitions[current_state][action]:
            if possible_new_state == new_state:
                return probability
        return 0

    def get_belief_state_reward(self, belief_state):
        """
        Get the reward for a
        given belief state 
        """
        state_reward_sum = 0
        
        for state in belief_state:
            state_reward = self.grid_pomdb.rewards[state]
            weighted_state_reward = state_reward * belief_state[state]
            state_reward_sum = state_reward_sum + weighted_state_reward
        
        return state_reward_sum

    def get_belief_state_utility(self, belief_state):
        pass

    def get_utility_of_state(self, state, visited_states):
        """
        The Bellman equation
        """
        reward_current_state = self.grid_pomdb.rewards[state]
        
        best_follow_up_state_utility = 0
        for action in self.grid_pomdb.transitions[state]:
            action_state_transitions = self.grid_pomdb.transitions[state][action]
            for state_transition in action_state_transitions:
                possible_state_probability = state_transition[0]
                possible_state = state_transition[1]

                if possible_state in visited_states:
                    continue
                else:
                    visited_states.append(possible_state)

                possible_state_utility = self.get_utility_of_state(possible_state, visited_states)
                weighted_possible_state_utility = possible_state_utility * possible_state_probability

                if weighted_possible_state_utility > best_follow_up_state_utility:
                    best_follow_up_state_utility = weighted_possible_state_utility
                    
        # return reward of state + (result of above)
        return reward_current_state + best_follow_up_state_utility

ddn = DynamicDecisionNetwork(20)
ddn.solve_grid()