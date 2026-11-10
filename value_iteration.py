from environment.cliff import CliffEnv

# gen_obs(): row = height - 2 - y = 4 - y, col = x - 1 --> row 0 is bottom
# cliff: x = 2~11, y = 4  -> row 0, col 1~10
# goal: x = 12, y = 4 --> row 0, col 11
ROWS, COLS = 4, 12
GOAL = (0, 11)
CLIFF = {(0, col) for col in range(1, 11)}

# actions 0,1,2,3 = up, down, left, right
# up decreases y in env, but row = 4 - y gets bigger --> do (+1, 0) isstead of (-1, 0)
DELTA = {0: (+1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, +1)}


def actions_of(env, state):
    # actions from 'state' that stays within borders
    # just testing, no actual movement
    row, col = state
    env.agent_pos = (col + 1, env.height - 2 - row)
    return env.get_available_actions()


def step_model(state, action):
    # (state, action) -> (next_state, reward): just testing, no actual movement
    # env.step() would actually move
    row, col = state
    d_row, d_col = DELTA[action]
    next_state = (row + d_row, col + d_col)
    reward = -100 if next_state in CLIFF else -1
    return next_state, reward


if __name__ == "__main__":
    env = CliffEnv()
    env.reset() # must do to initialise field

    print("actions in (0, 0):", actions_of(env, (0, 0)))
    print("right step from (0, 0):", step_model((0, 0), 3))
    print("up step from (0, 0):", step_model((0, 0), 0))
    print("actions in (3, 5):", actions_of(env, (3, 5)))
