import random
import json
import dataclasses


@dataclasses.dataclass
class GameState:
    # Distance to good food
    distance: tuple
    # Tuple of (0 or 1 or NA) and (2 or 3 or NA) to represent where the food is in relation to the snake head
    # Values of directions correspond to the action keys (see Agent's "actions" field)
    # 0 - Food Left of Snake, 1 - Food Right of Snake, NA - Food in same X as snake head
    # 2 - Food Above Snake, 3 - Food Below Snake, Food in same Y as snake head
    position: tuple
    # String of 0, 1 and 2 describing whether surrounding squares are free tiles, walls/off-screen, or bad fruit
    surroundings: str
    # Position of good Food
    food: tuple
    # Position of bad Food
    badfood: tuple

# =================================================================================================


class Agent(object):
    def __init__(self, display_width, display_height, block_size):
        # Game parameters
        self.display_width = display_width
        self.display_height = display_height
        self.block_size = block_size

        # Learning parameters
        self.epsilon = 0.1
        self.lr = 0.75
        self.discount = .5

        # All possible rewards stored in array for easy changing
        # Order: Death, Eating Food, Eating Bad Food, Moving Closer to Food, Moving Away from Food
        self.reward_array = [-100, 50, -25, 15, -15]
        #[-1, 1, -1, 1, -1]
        #[-3, 2, -1.5, 1, -1]
        #[-100, 50, -25, 15, -15]

        # State/Action history
        self.qvalues = self.LoadQvalues()
        self.history = []

        # Action space
        self.actions = {
            0: 'L',
            1: 'R',
            2: 'U',
            3: 'D'
        }

    # Reset history of agent
    def Reset(self):
        self.history = []

    # Load QVals from the json file
    def LoadQvalues(self, path="qvalues.json"):
        with open(path, "r") as f:
            qvalues = json.load(f)
        return qvalues

    # Save the current QVals to the json file
    def SaveQvalues(self, path="qvalues.json"):
        with open(path, "w") as f:
            json.dump(self.qvalues, f)

    # Return the correct action to take based on the current state and QVals
    def act(self, snake, food, bad_food, walls):
        # Get the current state of the game
        state = self._GetState(snake, food, bad_food, walls)

        # Epsilon
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            # Choose random direction to go
            action_key = random.choices(list(self.actions.keys()))[0]
        else:
            # Use the QVals to determine the best direction to go
            state_scores = self.qvalues[self._GetStateStr(state)]
            action_key = state_scores.index(max(state_scores))
        action_val = self.actions[action_key]

        # Remember the actions it took at each state
        self.history.append({
            'state': state,
            'action': action_key
        })
        return action_val

    # Updating the QVals, done in batches to improve performance
    def UpdateQValues(self, reason):
        history = self.history[::-1]
        for i, h in enumerate(history[:-1]):
            if reason:  # Snake Died -> Negative reward
                sN = history[0]['state']
                aN = history[0]['action']
                state_str = self._GetStateStr(sN)
                # Reward for death
                reward = self.reward_array[0]
                # Bellman equation - there is no future state since game is over
                self.qvalues[state_str][aN] = (
                    1-self.lr) * self.qvalues[state_str][aN] + self.lr * reward
                reason = None
            else:
                s1 = h['state']  # current state
                s0 = history[i+1]['state']  # previous state
                a0 = history[i+1]['action']  # action taken at previous state

                x1 = s0.distance[0]  # x distance at current state
                y1 = s0.distance[1]  # y distance at current state

                x2 = s1.distance[0]  # x distance at previous state
                y2 = s1.distance[1]  # y distance at previous state

                if s0.food != s1.food:  # Snake ate a food, positive reward
                    # Reward for eating good food
                    reward = self.reward_array[1]
                elif s0.badfood != s1.badfood:
                    # Reward for eating bad food
                    reward = self.reward_array[2]
                # Snake is closer to the food, positive reward
                elif (abs(x1) > abs(x2) or abs(y1) > abs(y2)):
                    # Reward for moving closer
                    reward = self.reward_array[3]
                else:
                    # Reward for moving away
                    # Snake is further from the food, negative reward
                    reward = self.reward_array[4]

                state_str = self._GetStateStr(s0)
                new_state_str = self._GetStateStr(s1)
                self.qvalues[state_str][a0] = (1-self.lr) * (self.qvalues[state_str][a0]) + self.lr * (
                    reward + self.discount*max(self.qvalues[new_state_str]))  # Bellman equation

    # Get the State class from the given information
    def _GetState(self, snake, food, bad_food, walls):
        # Get snake head position
        snake_head = snake[-1]
        # Calculate distance to good food
        dist_x = food[0] - snake_head[0]
        dist_y = food[1] - snake_head[1]

        # Get position values
        if dist_x > 0:
            pos_x = '1'  # Food is to the right of the snake
        elif dist_x < 0:
            pos_x = '0'  # Food is to the left of the snake
        else:
            pos_x = 'NA'  # Food and snake are on the same X file

        if dist_y > 0:
            pos_y = '3'  # Food is below snake
        elif dist_y < 0:
            pos_y = '2'  # Food is above snake
        else:
            pos_y = 'NA'  # Food and snake are on the same Y file

        # Get surrounding values
        sqs = [
            (snake_head[0] - self.block_size, snake_head[1]),
            (snake_head[0] + self.block_size, snake_head[1]),
            (snake_head[0], snake_head[1] - self.block_size),
            (snake_head[0], snake_head[1] + self.block_size),
        ]

        surrounding_list = []
        # Check each of the 4 surrounding squares
        for sq in sqs:
            # Surrounding square is off screen
            if sq[0] < 0 or sq[1] < 0 or sq[0] >= self.display_width or sq[1] >= self.display_height:
                surrounding_list.append('1')
            # Surrounding square is a bad food
            elif sq == bad_food:
                surrounding_list.append('1')
            # Surrounding square is a wall or the snake itself
            elif sq in snake[:-1] or sq in walls:
                surrounding_list.append('1')
            # Surrounding square is an open tile
            else:
                surrounding_list.append('0')
        surroundings = ''.join(surrounding_list)

        return GameState((dist_x, dist_y), (pos_x, pos_y), surroundings, food, bad_food)

    # Translate game state into a string
    # String is a three element tuple of left or right of food, up or down of food, and surroundings
    def _GetStateStr(self, state):
        return str((state.position[0], state.position[1], state.surroundings))
