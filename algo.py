from collections import OrderedDict
import numpy as np
from state import next_state, solved_state


class Node:
    def __init__(self, parent, action, cost, state, location=None):
        self.parent = parent
        self.action = action
        self.cost = cost
        self.state = state
        if location is not None:
            self.location = location

def hash_fn(state, cost=None):
    if cost is None:
        return hash(state.data.tobytes())
    else:
        state_flattened = state.flatten()
        state_action = np.zeros(state_flattened.shape[0]+1)
        state_action[:state_flattened.shape[0]] = state_flattened
        state_action[-1] = cost
        return hash(state_action.data.tobytes())

def get_hashed_goal_state():
    hashed_goal_states = {}
    state = solved_state()
    hashed_goal_states[hash_fn(state)] = state
    return hashed_goal_states

def backtrack(final_node):
    action_sequence = []
    current_node = final_node
    while current_node is not None:
        action_sequence.append(current_node.action)
        current_node = current_node.parent
    action_sequence = list(reversed(action_sequence))[1:]
    return action_sequence

def bfs(init_state, hashed_goal_states):
    explored_dict = {}
    frontier_dict = OrderedDict()

    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash_fn(initial_node.state)

    frontier_dict[init_hashed_state] = initial_node
    max_depth = 0

    while True:
        if len(frontier_dict) == 0:
            return None, expanded_num, explored_num
        
        hashed_state, node = frontier_dict.popitem(last=False)
        explored_dict[hashed_state] = node

        explored_num += 1

        if node.cost > max_depth:
            max_depth = node.cost
        
        if hashed_state in hashed_goal_states: 
            return node, expanded_num, explored_num
        
        for i in range(1, 12+1):
            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)
            if new_hashed_state in explored_dict or new_hashed_state in frontier_dict:
                continue
            new_node = Node(node, i, node.cost + 1, new_state)
            frontier_dict[new_hashed_state] = new_node
            expanded_num += 1

def dls(init_state, hashed_goal_states, limit):

    explored_dict = {}
    frontier_dict = OrderedDict()

    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash_fn(initial_node.state)

    frontier_dict[init_hashed_state] = initial_node

    max_depth = 0

    while True:

        if len(frontier_dict) == 0:
            return None, expanded_num, explored_num

        hashed_state, node = frontier_dict.popitem(last=True)

        explored_dict[hashed_state] = node

        explored_num += 1  # one node explored

        # printing max depth
        if node.cost > max_depth:
            max_depth = node.cost

        # checking if it is a goal state
        if hashed_state in hashed_goal_states: 
            return node, expanded_num, explored_num

        if node.cost == limit:
            continue

        # checking every possible move
        for i in range(1, 12+1):
            
            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)
            new_node = Node(node, i, node.cost + 1, new_state)

            if new_hashed_state in explored_dict:
                existing_node = explored_dict[new_hashed_state]
                if existing_node.cost <= new_node.cost:
                    continue

            if new_hashed_state in frontier_dict:
                existing_node = frontier_dict[new_hashed_state]
                if existing_node.cost <= new_node.cost:
                    continue

            frontier_dict[new_hashed_state] = new_node
            expanded_num += 1  # one node expanded


def ids(init_state, hashed_goal_states, max_limit):
    for depth in range(1, max_limit+1):
        final_node, expanded_num, explored_num = dls(init_state, hashed_goal_states, limit=depth)
        if final_node is not None:
            return final_node, expanded_num, explored_num
    return None, expanded_num, explored_num

def solve(init_state, method):
    
    if method == 'BFS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = bfs(init_state, hashed_goal_states)
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'DLS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = dls(init_state, hashed_goal_states, limit=7)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'IDS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = ids(init_state, hashed_goal_states, max_limit=9)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence

    return []