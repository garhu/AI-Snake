# AI Snake

AI Snake is an AI agent that uses a Q-learning algorithm to play the game. To test the decision making process of the agent, new elements to the game were introduced: walls and bad fruit. Walls appear after each fruit is picked up and collision with the walls result in a game over condition. Bad fruit appear similarly to normal fruit, but they decrease the score. These new obstacles allowed for analysis of the agent's decision-making capabilities in a dynamically growing maze.

## How to Run

```bash
python3 -m venv snake-env
source snake-env/bin/activate
pip install -r requirements.txt
python InitializeQVals.py
python AI_Snake.py
```

Feel free to edit AI_Snake.py to change draw_display and snake_speed parameters
