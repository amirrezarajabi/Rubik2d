from collections import OrderedDict
import numpy as np
from state import next_state, solved_state
from location import next_location, solved_location
import heapq

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

def bi_backtrack(s_final_node, g_final_node):

    action_sequence = []

    current = s_final_node
    while current.parent is not None:
        action_sequence.insert(0, current.action)
        current = current.parent

    current = g_final_node
    reverse_dict = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12,
                    7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
    while current.parent is not None:
        action_sequence.append(reverse_dict[current.action])
        current = current.parent

    return action_sequence

def is_found(g_frontier_dict, s_frontier_dict):
    hashed_common_state = None
    for i in s_frontier_dict:
        if i in g_frontier_dict:
            hashed_common_state = i
            break

    # if any state if found
    if hashed_common_state is not None:
        g_final_node = g_frontier_dict[hashed_common_state]
        s_final_node = s_frontier_dict[hashed_common_state]
        return s_final_node, g_final_node
    else:
        return None, None

def one_step_bfs(frontier_dict, explored_set, depth):

    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    while True:

        if len(frontier_dict) == 0:
            return
        
        hashed_state = next(iter(frontier_dict))
        node = frontier_dict[hashed_state]

        if node.cost != depth:
            return expanded_num, explored_num
        
        explored_num += 1
        
        explored_set.add(hashed_state)

        frontier_dict.pop(hashed_state)

        for i in range(1, 12+1):

            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)

            if new_hashed_state in explored_set or new_hashed_state in frontier_dict:
                continue

            new_node = Node(node, i, node.cost + 1, new_state)
            frontier_dict[new_hashed_state] = new_node

            expanded_num += 1  # one node expanded

def bibfs(init_state, hashed_goal_states):
    
    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    s_explored_set = set()
    g_explored_set = set()

    s_frontier_dict = OrderedDict()
    g_frontier_dict = OrderedDict()

    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash_fn(initial_node.state)

    s_frontier_dict[init_hashed_state] = initial_node

    for g in hashed_goal_states.keys():
        goal_state = hashed_goal_states[g]
        goal_node = Node(None, None, 0, goal_state)
        g_frontier_dict[g] = goal_node

    s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
    if s_final_node is not None:
        return s_final_node, g_final_node, expanded_num, explored_num

    depth = 0
    while True:


        if len(s_frontier_dict) == 0 or len(g_frontier_dict) == 0:
            return None, None, expanded_num, explored_num
        
        expanded_num_added, explored_num_added = one_step_bfs(s_frontier_dict, s_explored_set, depth)
        expanded_num += expanded_num_added
        explored_num += explored_num_added

        s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
        if s_final_node is not None:
            return s_final_node, g_final_node, expanded_num, explored_num

        expanded_num_added, explored_num_added = one_step_bfs(g_frontier_dict, g_explored_set, depth)
        expanded_num += expanded_num_added
        explored_num += explored_num_added

        s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
        if s_final_node is not None:
            return s_final_node, g_final_node, expanded_num, explored_num

        depth += 1

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

matrix = np.zeros((8, 8), dtype=np.uint8)
matrix[:4, :4] = 1
matrix[:4, 4:] = 2
matrix[4:, 4:] = 1
matrix[4:, :4] = 2
np.fill_diagonal(matrix, 0)
np.fill_diagonal(np.fliplr(matrix), 3)
np.fill_diagonal(matrix[:4, 4:], 1)
np.fill_diagonal(matrix[4:, :4], 1)
np.fill_diagonal(np.fliplr(matrix[:4, :4]), 2)
np.fill_diagonal(np.fliplr(matrix[4:, 4:]), 2)

def heuristic(location, is_ucs):  # heuristic function
    
    # h(node) = 0 -> UCS
    if is_ucs: return 0

    select_idx = location.flatten() - 1
    output_array = np.choose(select_idx, matrix)
    h = np.sum(output_array) / 4
    return h

def a_star(init_state, init_location, hashed_goal_states, is_ucs=False):

    all_expanded_set = set()
    
    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    # creating the initial node
    initial_node = Node(None, None, 0, init_state, init_location)
    init_hashed_state_cost = hash_fn(initial_node.state, initial_node.cost)
    all_expanded_set.add(init_hashed_state_cost)

    # add initial node to frontier
    frontier_pq = []
    heapq.heappush(frontier_pq, (heuristic(initial_node.location, is_ucs=is_ucs) + initial_node.cost, id(initial_node), initial_node))

    max_depth = 0

    while True:

        if len(frontier_pq) == 0:
            return None, None, None
        
        priority, node_id, node = heapq.heappop(frontier_pq)

        hashed_state = hash_fn(node.state)

        explored_num += 1  # one node explored

        # printing max depth
        if node.cost > max_depth:
            max_depth = node.cost

        # checking if it is a goal state
        if hashed_state in hashed_goal_states: 
            return node, expanded_num, explored_num
        
        for i in range(1, 12+1):
            
            new_state = next_state(node.state, action=i)
            new_location = next_location(node.location, action=i)
            new_hashed_state_cost = hash_fn(new_state, node.cost+1)

            if node.cost+1 > 14:
                continue

            if new_hashed_state_cost in all_expanded_set:
                continue

            all_expanded_set.add(new_hashed_state_cost)

            new_node = Node(node, i, node.cost + 1, new_state, new_location)
            heapq.heappush(frontier_pq, (heuristic(new_node.location, is_ucs=is_ucs) + new_node.cost, id(new_node), new_node))

            expanded_num += 1  # one node expanded

def ucs(init_state, init_location, hashed_goal_states):
    return a_star(init_state, init_location, hashed_goal_states, is_ucs=True) 

def solve(init_state, init_location, method):
    
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
    
    elif method == 'BiBFS':
        hashed_goal_states = get_hashed_goal_state()
        s_final_node, g_final_node, expanded_num, explored_num = bibfs(init_state, hashed_goal_states)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = bi_backtrack(s_final_node, g_final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'A*':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = a_star(
            init_state, init_location, hashed_goal_states)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'UCS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = ucs(
            init_state, init_location, hashed_goal_states)        
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