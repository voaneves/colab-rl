#!/usr/bin/env python

"""SnakeGame: A simple and fun exploration, meant to be used by AI algorithms.
"""

from sys import exit # To close the window when the game is over
from os import environ # To center the game window the best possible
import random # Random numbers used for the food
import logging # Logging function for movements and errors
from itertools import tee # For the color gradient on snake
# !pip install pygame # Jupyter Notebook
import pygame # This is the engine used in the game
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

actions = {0:'LEFT', 1:'RIGHT', 2:'UP', 3:'DOWN', 4:'idle'}
forbidden_moves = [(0, 1), (1, 0), (2, 3), (3, 2)]

class GlobalVariables:
    """Global variables to be used while drawing and moving the snake game.

    Attributes:
        BLOCK_SIZE: The size in pixels of a block.
        HEAD_COLOR: Color of the head.
        BODY_COLOR: Color of the body.
        FOOD_COLOR: Color of the food.
        GAME_SPEED: Speed in ticks of the game. The higher the faster.
    """
    def __init__(self):
        """Initialize all global variables."""
        self.BOARD_SIZE = 10
        self.BLOCK_SIZE = 20
        self.HEAD_COLOR = (0, 0, 0)
        self.BODY_COLOR = (0, 200, 0)
        self.FOOD_COLOR = (200, 0, 0)
        self.GAME_SPEED = 24

        if self.BOARD_SIZE > 50:
            logger.warning('WARNING: BOARD IS TOO BIG, IT MAY RUN SLOWER.')


class Snake:
    """Player (snake) class which initializes head, body and board.

    The body attribute represents a list of positions of the body, which are in-
    cremented when moving/eating on the position [0]. The orientation represents
    where the snake is looking at (head) and collisions happen when any element
    is superposed with the head.

    Attributes:
        head: The head of the snake, located according to the board size.
        body: Starts with 3 parts and grows when food is eaten.
        orientation: Current orientation where head is pointing.
    """
    def __init__(self):
        """Inits Snake with 3 body parts (one is the head) and pointing right"""
        self.head = [int(var.BOARD_SIZE / 4), int(var.BOARD_SIZE / 4)]
        self.body = [[self.head[0], self.head[1]],
                     [self.head[0] - 1, self.head[1]],
                     [self.head[0] - 2, self.head[1]]]
        self.previous_action = 1
        self.length = 3

    def move(self, action, food_pos):
        """According to orientation, move 1 block. If the head is not positioned
        on food, pop a body part. Else (food), return without popping."""
        if action == 4 or (action, self.previous_action) in forbidden_moves:
            action = self.previous_action
        else:
            self.previous_action = action

        if action == 0:
            self.head[0] -= 1
        elif action == 1:
            self.head[0] += 1
        elif action == 2:
            self.head[1] -= 1
        elif action == 3:
            self.head[1] += 1

        self.body.insert(0, list(self.head))

        if self.head == food_pos:
            logger.info('EVENT: FOOD EATEN')
            self.length = len(self.body)

            return True
        else:
            self.body.pop()

            return False

    def check_collision(self):
        """Check wether any collisions happened with the wall or body and re-
        turn."""
        if self.head[0] > (var.BOARD_SIZE - 1) or self.head[0] < 0:
            logger.info('EVENT: WALL COLLISION')

            return True
        elif self.head[1] > (var.BOARD_SIZE - 1) or self.head[1] < 0:
            logger.info('EVENT: WALL COLLISION')

            return True
        elif self.head in self.body[1:]:
            logger.info('EVENT: BODY COLLISION')

            return True

        return False

    def return_body(self):
        """Return the whole body."""
        return self.body


class FoodGenerator:
    """Generate and keep track of food.

    Attributes:
        pos: Current position of food.
        is_food_on_screen: Flag for existence of food.
    """
    def __init__(self, body):
        """Initialize a food piece and set existence flag."""
        self.is_food_on_screen = False
        self.pos = self.generate_food(body)

    def generate_food(self, body):
        """Generate food and verify if it's on a valid place."""
        if not self.is_food_on_screen:
            while True:
                food = [int((var.BOARD_SIZE - 1) * random.random()),
                        int((var.BOARD_SIZE - 1) * random.random())]

                if food in body:
                    continue
                else:
                    self.pos = food
                    break

            logger.info('EVENT: FOOD APPEARED')
            self.is_food_on_screen = True

        return self.pos

    def set_food_on_screen(self, bool_value):
        """Set flag for existence (or not) of food."""
        self.is_food_on_screen = bool_value


class Game:
    """Hold the game window and functions.

    Attributes:
        window: pygame window to show the game.
        fps: Define Clock and ticks in which the game will be displayed.
        snake: The actual snake who is going to be played.
        food_generator: Generator of food which responds to the snake.
        food_pos: Position of the food on the board.
        game_over: Flag for game_over.
    """
    def __init__(self, board_size = 10):
        """Initialize window, fps and score."""
        var.BOARD_SIZE = board_size
        self.reset()

    def reset(self):
        self.step = 0
        self.snake = Snake()
        self.food_generator = FoodGenerator(self.snake.body)
        self.food_pos = self.food_generator.pos
        self.scored = False
        self.game_over = False

    def create_window(self):
        self.window = pygame.display.set_mode((var.BOARD_SIZE * var.BLOCK_SIZE,\
                                               var.BOARD_SIZE * var.BLOCK_SIZE))
        self.fps = pygame.time.Clock()

    def start(self):
        """Create some wait time before the actual drawing of the game."""
        for i in range(3):
            pygame.display.set_caption("SNAKE GAME  |  Game starts in " +\
                                       str(3 - i) + " second(s) ...")
            pygame.time.wait(1000)
        logger.info('EVENT: GAME START')

    def over(self):
        """If collision with wall or body, end the game."""
        pygame.display.set_caption("SNAKE GAME  |  Score: "
                            + str(self.snake.length - 3)
                            + "  |  GAME OVER. Press any Q or ESC to quit ...")
        logger.info('EVENT: GAME OVER')

        while True:
            keys = pygame.key.get_pressed()
            pygame.event.pump()

            if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
                break

        pygame.quit()
        exit()

    def is_won(self):
        return self.snake.length > 3

    def generate_food(self):
        return self.food_generator.generate_food(self.snake.body)

    def handle_input(self, previous_action):
        """After getting current pressed keys, handle important cases."""
        keys = pygame.key.get_pressed()
        pygame.event.pump()

        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
            self.over()
        elif keys[pygame.K_LEFT]:
            logger.info('ACTION: KEY PRESSED: LEFT')
            return 0
        elif keys[pygame.K_RIGHT]:
            logger.info('ACTION: KEY PRESSED: RIGHT')
            return 1
        elif keys[pygame.K_UP]:
            logger.info('ACTION: KEY PRESSED: UP')
            return 2
        elif keys[pygame.K_DOWN]:
            logger.info('ACTION: KEY PRESSED: DOWN')
            return 3
        else:
            return previous_action

    def state(self):
        """Create a matrix of the current state of the game."""
        body = self.snake.return_body()
        canvas = np.zeros((var.BOARD_SIZE, var.BOARD_SIZE))

        for part in body:
            canvas[part[0], part[1]] = 1.

        canvas[self.food_pos[0], self.food_pos[1]] = .5

        return canvas

    def play(self, action, player):
        """Move the snake to the direction, eat and check collision."""
        assert action in range(5), "Invalid action."

        self.scored = False
        self.step += 1
        self.food_pos = self.generate_food()

        if self.snake.move(action, self.food_pos):
            self.scored = True
            self.food_generator.set_food_on_screen(False)

        if self.snake.check_collision():
            self.game_over = True

            if player == "HUMAN":
                self.over()

    def get_reward(self):
        """Return the current score. Can be used as the reward function."""
        if self.game_over:
            return -1
        elif self.scored:
            return self.snake.length

        return 0

    def gradient(self, colors, steps, components = 3):
        """Function to create RGB gradients given 2 colors and steps.

        If component is changed to 4, it does the same to RGBA colors."""
        def linear_gradient(start, finish, substeps):
            yield start
            for i in range(1, substeps):
                yield tuple([(start[j] + (float(i) / (substeps-1)) * (finish[j]\
                            - start[j])) for j in range(components)])

        def pairs(seq):
            a, b = tee(seq)
            next(b, None)
            return zip(a, b)

        result = []
        substeps = int(float(steps) / (len(colors) - 1))

        for a, b in pairs(colors):
            for c in linear_gradient(a, b, substeps):
                result.append(c)

        return result

    def draw(self, color_list):
        """Draw the game, the snake and the food using pygame."""
        self.window.fill(pygame.Color(225, 225, 225))

        for part, color in zip(self.snake.body, color_list):
            pygame.draw.rect(self.window, color, pygame.Rect(part[0] *\
                        var.BLOCK_SIZE, part[1] * var.BLOCK_SIZE, \
                        var.BLOCK_SIZE, var.BLOCK_SIZE))

        pygame.draw.rect(self.window, var.FOOD_COLOR,\
                         pygame.Rect(self.food_pos[0] * var.BLOCK_SIZE,\
                         self.food_pos[1] * var.BLOCK_SIZE, var.BLOCK_SIZE,\
                         var.BLOCK_SIZE))

        pygame.display.set_caption("SNAKE GAME  |  Score: "
                                    + str(self.snake.length - 3))
        pygame.display.update()
        self.fps.tick(var.GAME_SPEED)


def main():
    """The main function where the game will be executed."""
    # Setup basic configurations for logging in this module
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
    game = Game()
    game.create_window()
    game.start()

    # The main loop, it pump key_presses and update the board every tick.
    previous_size = game.snake.length # Initial size of the snake
    current_size = 3 # Initial size of the snake
    color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
                               previous_size)

    # Main loop, where the snake keeps going each tick. It generate food, check
    # collisions and draw.
    while True:
        action = game.handle_input(game.snake.previous_action)
        game.play(action, "HUMAN")
        game.draw(color_list)

        current_size = game.snake.length # Update the body size

        if current_size > previous_size:
            color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
                                       game.snake.length)

        previous_size = current_size

var = GlobalVariables() # Initializing GlobalVariables
logger = logging.getLogger(__name__) # Setting logger
environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window

if __name__ == '__main__':
    main() # Execute game! Let's play ;)
