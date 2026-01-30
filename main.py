import numpy as np
import argparse
import time
from state import solved_state, next_state
from algo import solve


if __name__ == '__main__':

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase', type=str, default=None)
    parser.add_argument('--method', type=str, default='BFS')
    args = parser.parse_args()


    # initializing state
    state = solved_state()

    # scramble
    if args.testcase is None:
        scramble_sequence = np.random.randint(1, 12+1, np.random.randint(10, 30))
    else:
        f = open(args.testcase, 'r')
        scramble_sequence = list(map(int, f.readline().split()))
    
    # calculate the state
    for a in scramble_sequence:
        state = next_state(state, action=a)

    # solve rubik
    print('------------------ START ------------------')
    print('SOLVING...')
    start_time = time.time()
    solve_sequence = solve(state, method=args.method)
    print('actions:', solve_sequence)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'SOLVE FINISHED In {elapsed_time:.5f}s.')

