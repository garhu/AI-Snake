import pygame
import random
import Agent
import matplotlib.pyplot as plt

pygame.init()

# Initialize colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
grey = (169, 169, 169)
green = (0, 255, 0)
blue = (50, 153, 213)

# Initialize width and height
dis_width = 400
dis_height = 400

# Draw the display or not?
draw_display = True

# Set up display
if draw_display:
    dis = pygame.display.set_mode((dis_width, dis_height))
    pygame.display.set_caption('Snake')
    pygame.event.get()

# Set up clock
clock = pygame.time.Clock()

# Set how much the snake moves per tick
# Essentially the size of the snake
snake_block = 20

# Set how fast the snake moves (no GUI recommended speed = 20000)
snake_speed = 10

# Set score font for the game
score_font = pygame.font.SysFont("bahnschrift", 15)

# Set upper limit to end trials if they loop forever
max_move_count = 1000

# =================================================================================================


def your_score(score):
    # Drawing score
    value = score_font.render("Score: " + str(score), True, white)
    dis.blit(value, [10, 10])


def our_snake(snake_block, snake_list):
    # Drawing snake
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])


def get_new_position(object_list):
    # Get new position, given a list of objects to avoid
    while True:
        position = (round(random.randrange(0, dis_width - snake_block) / snake_block) *
                    snake_block, round(random.randrange(0, dis_height - snake_block) / snake_block) * snake_block)
        if position not in object_list:
            return position

# =================================================================================================


def gameLoop():
    # Initial game conditions
    game_over = False
    reason = None

    # Initialize the starting position of the snake
    x1 = dis_width / 2
    y1 = dis_height / 2

    # Initialize direction of snake
    x1_change = 0
    y1_change = 0

    # Initialize snake
    snake_list = [(x1, y1)]
    length_of_snake = 1

    # Initialize walls
    wall_list = []

    # Initialize foods
    foodx, foody = get_new_position(snake_list)
    bad_foodx, bad_foody = get_new_position(snake_list + [(foodx, foody)])

    # Initialize score
    score = 0

    # Move count
    move_count = 0

    while not game_over:
        # Agent moves snake
        action = agent.act(snake_list, (foodx, foody),
                           (bad_foodx, bad_foody), wall_list)
        if action == "L":
            x1_change = -snake_block
            y1_change = 0
        elif action == "R":
            x1_change = snake_block
            y1_change = 0
        elif action == "U":
            x1_change = 0
            y1_change = -snake_block
        elif action == "D":
            x1_change = 0
            y1_change = snake_block

        # Update snake position
        x1 += x1_change
        y1 += y1_change
        snake_head = (x1, y1)
        snake_list.append(snake_head)

        move_count += 1
        if move_count > max_move_count:
            game_over, reason = True, "Loop"

        # Game over conditions (off-screen)
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_over, reason = True, "Off-Screen"
        # Check if bad food was found
        if x1 == bad_foodx and y1 == bad_foody:
            bad_foodx, bad_foody = get_new_position(
                snake_list + wall_list + [(foodx, foody)])
            score -= 2
        # If you ate a bad food at 0 score, it is game over
        if score < 0:
            game_over, reason = True, "Bad Food"
        # Self colision
        for x in snake_list[:-1]:
            if x == snake_head:
                game_over, reason = True, "Self-Collision"
        # Has collided with wall, then it is game over
        if snake_head in wall_list:
            game_over, reason = True, "Wall-Collision"

        # Check if food was found
        if x1 == foodx and y1 == foody:
            wall_list.append(get_new_position(wall_list + snake_list))
            foodx, foody = get_new_position(snake_list + wall_list)
            bad_foodx, bad_foody = get_new_position(
                snake_list + wall_list + [(foodx, foody)])
            length_of_snake += 1
            score += 1

        # Delete tail
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        if draw_display:
            # Draw background
            dis.fill(blue)
            # Draw good food
            pygame.draw.rect(
                dis, green, [foodx, foody, snake_block, snake_block])
            # Draw bad food
            pygame.draw.rect(
                dis, red, [bad_foodx, bad_foody, snake_block, snake_block])
            # Draw walls
            for wall in wall_list:
                pygame.draw.rect(
                    dis, grey, [wall[0], wall[1], snake_block, snake_block])

            # Draw snake and score
            our_snake(snake_block, snake_list)
            your_score(score)

            pygame.display.update()

        # Update Q Table
        agent.UpdateQValues(reason)

        clock.tick(snake_speed)

    return score, reason

# =================================================================================================


# How often Q Vals are saved and how many times they are saved before plotting
batch_size = 100
total_batches = 15

# Counter for games and batches
game_count = 1
batch_count = 0

# Initialize Agent
agent = Agent.Agent(dis_width, dis_height, snake_block)

# Initialize results array/dictionary
result_array = []
reasons_dict = {"Loop": 0, "Self-Collision": 0, "Off-Screen": 0, "Bad Food": 0,
                "Self-Collision": 0, "Wall-Collision": 0}  # This is reset after every batch

# Checks whether plot has been displayed or not
plotted = False

# =================================================================================================

while True:
    agent.Reset()
    # Setting epsilon value
    # =====================================================
    #agent.epsilon = 0.1 / (batch_count + 1)
    # =====================================================
    if (batch_count < 2):
        agent.epsilon = 0.1
    else:
        agent.epsilon = 0
    # =====================================================
    # Run through game
    score, reason = gameLoop()
    # Print out results for each game
    # print(f"Games: {game_count}; Score: {score}; Reason: {reason}")
    # Output results of each game to console to monitor as agent is training
    # Increment counter and save results
    reasons_dict[reason] += 1
    result_array += [score]
    game_count += 1
    # Save qvalues every qvalue_dump_n games and display current total of deaths per reason
    if game_count % batch_size == 0:
        print("Saving Qvals for Batch", batch_count)
        agent.SaveQvalues()
        print("Reasons:", reasons_dict)
        reasons_dict = {"Loop": 0, "Self-Collision": 0, "Off-Screen": 0,
                        "Bad Food": 0, "Self-Collision": 0, "Wall-Collision": 0}
        batch_count += 1
    # Plot a windowed average of scores at points in training
    if batch_count == total_batches and not plotted:
        plotted = True
        result_length = len(result_array)
        average_plot = []
        # How many trials are averaged together
        average_count = 50
        for result_index in range(result_length - average_count):
            sum = 0
            for i in range(average_count):
                sum += result_array[result_index + i]
            average_plot += [sum / average_count]

        # Plot
        plt.plot(average_plot)
        plt.xlabel("Current Game")
        plt.ylabel("Score")
        plt.suptitle("Moving Average Over Games While Training")
        plt.show()
        # Done training
        break
