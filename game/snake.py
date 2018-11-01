#!/usr/bin/env python

"""SnakeGame: A simple and fun exploration, meant to be used by AI algorithms.
"""

import sys # To close the window when the game is over
from os import environ, path # To center the game window the best possible
import random # Random numbers used for the food
import logging # Logging function for movements and errors
from itertools import tee # For the color gradient on snake
import pygame # This is the engine used in the game
import numpy as np

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

# Actions, options and forbidden moves
options = {'QUIT': 0, 'PLAY': 1, 'BENCHMARK': 2, 'LEADERBOARDS': 3, 'MENU': 4, 'ADD_LEADERBOARDS': 5}
relative_actions = {'LEFT': 0, 'FORWARD': 1, 'RIGHT': 2}
actions = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'IDLE': 4}
forbidden_moves = [(0, 1), (1, 0), (2, 3), (3, 2)]

# Types of point in the board
point_type = {'EMPTY': 0, 'FOOD': 1, 'BODY': 2, 'HEAD': 3, 'DANGEROUS': 4}

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
        self.BOARD_SIZE = 30
        self.BLOCK_SIZE = 20
        self.HEAD_COLOR = (0, 0, 0)
        self.BODY_COLOR = (0, 200, 0)
        self.FOOD_COLOR = (200, 0, 0)
        self.GAME_SPEED = 10
        self.BENCHMARK = 10

        if self.BOARD_SIZE > 50:
            logger.warning('WARNING: BOARD IS TOO BIG, IT MAY RUN SLOWER.')

class TextBlock:
    def __init__(self, text, pos, screen, scale = (1 / 12), type = "text"):
        self.type = type
        self.hovered = False
        self.text = text
        self.pos = pos
        self.screen = screen
        self.scale = scale
        self.set_rect()
        self.draw()

    def draw(self):
        self.set_rend()
        self.screen.blit(self.rend, self.rect)

    def set_rend(self):
        font = pygame.font.Font(resource_path("resources/fonts/freesansbold.ttf"),
                                int((var.BOARD_SIZE * var.BLOCK_SIZE) * self.scale))
        self.rend = font.render(self.text, True, self.get_color(),
                                self.get_background())

    def get_color(self):
        if self.type == "menu":
            if self.hovered:
                return pygame.Color(42, 42, 42)
            else:
                return pygame.Color(152, 152, 152)

        return pygame.Color(42, 42, 42)

    def get_background(self):
        if self.type == "menu":
            if self.hovered:
                return pygame.Color(152, 152, 152)

        return None

    def set_rect(self):
        self.set_rend()
        self.rect = self.rend.get_rect()
        self.rect.center = self.pos


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
        if action == actions['IDLE']\
            or (action, self.previous_action) in forbidden_moves:
            action = self.previous_action
        else:
            self.previous_action = action

        if action == actions['LEFT']:
            self.head[0] -= 1
        elif action == actions['RIGHT']:
            self.head[0] += 1
        elif action == actions['UP']:
            self.head[1] -= 1
        elif action == actions['DOWN']:
            self.head[1] += 1

        self.body.insert(0, list(self.head))

        if self.head == food_pos:
            logger.info('EVENT: FOOD EATEN')
            self.length = len(self.body)

            return True
        else:
            self.body.pop()

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
    def __init__(self, player, board_size = 30, local_state = False, relative_pos = False):
        """Initialize window, fps and score."""
        var.BOARD_SIZE = board_size
        self.local_state = local_state
        self.relative_pos = relative_pos
        self.player = player

        if player == "ROBOT":
            if self.relative_pos:
                self.nb_actions = 3
            else:
                self.nb_actions = 5

            self.reset_game()

    def reset_game(self):
        self.step = 0
        self.snake = Snake()
        self.food_generator = FoodGenerator(self.snake.body)
        self.food_pos = self.food_generator.pos
        self.scored = False
        self.game_over = False

    def create_window(self):
        pygame.init()

        flags = pygame.DOUBLEBUF
        self.window = pygame.display.set_mode((var.BOARD_SIZE * var.BLOCK_SIZE,\
                                               var.BOARD_SIZE * var.BLOCK_SIZE),
                                               flags)
        self.window.set_alpha(None)
        self.fps = pygame.time.Clock()

    def menu(self):
        pygame.display.set_caption("SNAKE GAME  | PLAY NOW!")

        img = pygame.image.load(resource_path("resources/images/snake_logo.png"))
        img = pygame.transform.scale(img, (var.BOARD_SIZE * var.BLOCK_SIZE, int(var.BOARD_SIZE * var.BLOCK_SIZE / 3)))

        self.screen_rect = self.window.get_rect()
        img_rect = img.get_rect()
        img_rect.center = self.screen_rect.center

        menu_options = [TextBlock(' PLAY GAME ', (self.screen_rect.centerx,
                                                  4 * self.screen_rect.centery / 10),
                                                  self.window, (1 / 12), "menu"),
                        TextBlock(' BENCHMARK ', (self.screen_rect.centerx,
                                                  6 * self.screen_rect.centery / 10),
                                                  self.window, (1 / 12), "menu"),
                        TextBlock(' LEADERBOARDS ', (self.screen_rect.centerx,
                                                     8 * self.screen_rect.centery / 10),
                                                     self.window, (1 / 12), "menu"),
                        TextBlock(' QUIT ', (self.screen_rect.centerx,
                                             10 * self.screen_rect.centery / 10),
                                             self.window, (1 / 12), "menu")]

        while True:
            pygame.event.pump()
            ev = pygame.event.get()

            self.window.fill(pygame.Color(225, 225, 225))

            for option in menu_options:
                option.draw()

                if option.rect.collidepoint(pygame.mouse.get_pos()):
                    option.hovered = True

                    if option == menu_options[0]:
                        for event in ev:
                            if event.type == pygame.MOUSEBUTTONUP:
                                return options['PLAY']
                    elif option == menu_options[1]:
                        for event in ev:
                            if event.type == pygame.MOUSEBUTTONUP:
                                return options['BENCHMARK']
                    elif option == menu_options[2]:
                        for event in ev:
                            if event.type == pygame.MOUSEBUTTONUP:
                                return options['LEADERBOARDS']
                    elif option == menu_options[3]:
                        for event in ev:
                            if event.type == pygame.MOUSEBUTTONUP:
                                return options['QUIT']
                else:
                    option.hovered = False

            self.window.blit(img, img_rect.bottomleft)
            pygame.display.update()

    def start_match(self):
        """Create some wait time before the actual drawing of the game."""
        for i in range(3):
            time = str(3 - i)
            self.window.fill(pygame.Color(225, 225, 225))

            # Game starts in 3, 2, 1
            text = [TextBlock('Game starts in', (self.screen_rect.centerx,
                                                 4 * self.screen_rect.centery / 10),
                                                 self.window, (1 / 10), "text"),
                    TextBlock(time, (self.screen_rect.centerx,
                                                 12 * self.screen_rect.centery / 10),
                                                 self.window, (1 / 1.5), "text")]

            for text_block in text:
                text_block.draw()

            pygame.display.update()
            pygame.display.set_caption("SNAKE GAME  |  Game starts in "
                                       + time + " second(s) ...")

            pygame.time.wait(1000)

        logger.info('EVENT: GAME START')

    def start(self):
        """Use menu to select the option/game mode."""
        opt = self.menu()
        running = True

        while running:
            if opt == options['QUIT']:
                pygame.quit()
                sys.exit()
            elif opt == options['PLAY']:
                self.select_speed()
                self.reset_game()
                self.start_match()
                score = self.single_player()
                opt = self.over(score)
            elif opt == options['BENCHMARK']:
                self.select_speed()
                score = []

                for i in range(var.BENCHMARK):
                    self.reset_game()
                    self.start_match()
                    score.append(self.single_player())

                opt = self.over(score)
            elif opt == options['LEADERBOARDS']:
                pass
            elif opt == options['ADD_LEADERBOARDS']:
                pass
            elif opt == options['MENU']:
                opt = self.menu()

    def over(self, score):
        """If collision with wall or body, end the game."""
        menu_options = [None] * 5

        menu_options[0] = TextBlock(' PLAY AGAIN ', (self.screen_rect.centerx,
                                                     4 * self.screen_rect.centery / 10),
                                                     self.window, (1 / 15), "menu")
        menu_options[1] = TextBlock(' GO TO MENU ', (self.screen_rect.centerx,
                                                     6 * self.screen_rect.centery / 10),
                                                     self.window, (1 / 15), "menu")
        menu_options[3] = TextBlock(' QUIT ', (self.screen_rect.centerx,
                                               10 * self.screen_rect.centery / 10),
                                               self.window, (1 / 15), "menu")

        if isinstance(score, int):
            text_score = 'SCORE: ' + str(score)

        else:
            text_score = 'MEAN SCORE: ' + str(sum(score) / var.BENCHMARK)

            menu_options[2] = TextBlock(' ADD TO LEADERBOARDS ', (self.screen_rect.centerx,
                                                                  8 * self.screen_rect.centery / 10),
                                                                  self.window, (1 / 15), "menu")

        pygame.display.set_caption("SNAKE GAME  | " + text_score
                                   + "  |  GAME OVER...")
        logger.info('EVENT: GAME OVER | FINAL ' + text_score)

        menu_options[4] = TextBlock(text_score, (self.screen_rect.centerx,
                                                 15 * self.screen_rect.centery / 10),
                                                 self.window, (1 / 10), "text")

        while True:
            pygame.event.pump()
            ev = pygame.event.get()

            # Game over screen
            self.window.fill(pygame.Color(225, 225, 225))

            for option in menu_options:
                if option is not None:
                    option.draw()

                    if option.rect.collidepoint(pygame.mouse.get_pos()):
                        option.hovered = True

                        if option == menu_options[0]:
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONUP:
                                    return options['PLAY']
                        elif option == menu_options[1]:
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONUP:
                                    return options['MENU']
                        elif option == menu_options[2]:
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONUP:
                                    return options['ADD_LEADERBOARDS']
                        elif option == menu_options[3]:
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONUP:
                                    pygame.quit()
                                    sys.exit()
                    else:
                        option.hovered = False

            pygame.display.update()

    def single_player(self):
        # The main loop, it pump key_presses and update the board every tick.
        previous_size = self.snake.length # Initial size of the snake
        current_size = previous_size # Initial size
        color_list = self.gradient([(42, 42, 42), (152, 152, 152)],\
                                   previous_size)

        # Main loop, where the snake keeps going each tick. It generate food, check
        # collisions and draw.
        while True:
            action = self.handle_input()

            if self.play(action):
                return current_size

            self.draw(color_list)
            current_size = self.snake.length # Update the body size

            if current_size > previous_size:
                color_list = self.gradient([(42, 42, 42), (152, 152, 152)],\
                                           current_size)

                previous_size = current_size

    def check_collision(self):
        """Check wether any collisions happened with the wall or body and re-
        turn."""
        if self.snake.head[0] > (var.BOARD_SIZE - 1) or self.snake.head[0] < 0:
            logger.info('EVENT: WALL COLLISION')

            return True
        elif self.snake.head[1] > (var.BOARD_SIZE - 1) or self.snake.head[1] < 0:
            logger.info('EVENT: WALL COLLISION')

            return True
        elif self.snake.head in self.snake.body[1:]:
            logger.info('EVENT: BODY COLLISION')

            return True

        return False

    def is_won(self):
        return self.snake.length > 3

    def generate_food(self):
        return self.food_generator.generate_food(self.snake.body)

    def handle_input(self):
        """After getting current pressed keys, handle important cases."""
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        keys = pygame.key.get_pressed()
        pygame.event.pump()

        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
            self.over(self.snake.length - 3)
        elif keys[pygame.K_LEFT]:
            logger.info('ACTION: KEY PRESSED: LEFT')
            return actions['LEFT']
        elif keys[pygame.K_RIGHT]:
            logger.info('ACTION: KEY PRESSED: RIGHT')
            return actions['RIGHT']
        elif keys[pygame.K_UP]:
            logger.info('ACTION: KEY PRESSED: UP')
            return actions['UP']
        elif keys[pygame.K_DOWN]:
            logger.info('ACTION: KEY PRESSED: DOWN')
            return actions['DOWN']
        else:
            return self.snake.previous_action

    def eval_local_safety(self, canvas, body):
        """Evaluate the safety of the head's possible next movements."""
        if (body[0][0] + 1) > (var.BOARD_SIZE - 1)\
            or ([body[0][0] + 1, body[0][1]]) in body[1:]:
            canvas[var.BOARD_SIZE - 1, 0] = point_type['DANGEROUS']
        if (body[0][0] - 1) < 0 or ([body[0][0] - 1, body[0][1]]) in body[1:]:
            canvas[var.BOARD_SIZE - 1, 1] = point_type['DANGEROUS']
        if (body[0][1] - 1) < 0 or ([body[0][0], body[0][1] - 1]) in body[1:]:
            canvas[var.BOARD_SIZE - 1, 2] = point_type['DANGEROUS']
        if (body[0][1] + 1) > (var.BOARD_SIZE - 1)\
            or ([body[0][0], body[0][1] + 1]) in body[1:]:
            canvas[var.BOARD_SIZE - 1, 3] = point_type['DANGEROUS']

        return canvas

    def state(self):
        """Create a matrix of the current state of the game."""
        canvas = np.zeros((var.BOARD_SIZE, var.BOARD_SIZE))

        if self.game_over:
            pass
        else:
            body = self.snake.return_body()

            for part in body:
                canvas[part[0], part[1]] = point_type['BODY']

            canvas[body[0][0], body[0][1]] = point_type['HEAD']

            if self.local_state:
                canvas = self.eval_local_safety(canvas, body)

            canvas[self.food_pos[0], self.food_pos[1]] = point_type['FOOD']

        return canvas

    def relative_to_absolute(self, action):
        if action == relative_actions['FORWARD']:
            action = self.snake.previous_action
        elif action == relative_actions['LEFT']:
            if self.snake.previous_action == actions['LEFT']:
                action = actions['DOWN']
            elif self.snake.previous_action == actions['RIGHT']:
                action = actions['UP']
            elif self.snake.previous_action == actions['UP']:
                action = actions['LEFT']
            else:
                action = actions['RIGHT']
        else:
            if self.snake.previous_action == actions['LEFT']:
                action = actions['UP']
            elif self.snake.previous_action == actions['RIGHT']:
                action = actions['DOWN']
            elif self.snake.previous_action == actions['UP']:
                action = actions['RIGHT']
            else:
                action = actions['LEFT']

        return action

    def play(self, action):
        """Move the snake to the direction, eat and check collision."""
        self.scored = False
        self.step += 1
        self.food_pos = self.generate_food()

        if self.relative_pos:
            action = self.relative_to_absolute(action)

        if self.snake.move(action, self.food_pos):
            self.scored = True
            self.food_generator.set_food_on_screen(False)

        if self.player == "HUMAN":
            if self.check_collision():
                return True
        elif self.check_collision() or self.step > 50 * self.snake.length:
            self.game_over = True

    def get_reward(self):
        """Return the current score. Can be used as the reward function."""
        if self.game_over:
            return -1
        elif self.scored:
            return self.snake.length

        return -0.005

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

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return path.join(sys._MEIPASS, relative_path)

    return path.join(path.dirname(path.realpath(__file__)), relative_path)

var = GlobalVariables() # Initializing GlobalVariables
logger = logging.getLogger(__name__) # Setting logger
environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window

if __name__ == '__main__':
    """The main function where the game will be executed."""
    # Setup basic configurations for logging in this module
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
    game = Game(player = "HUMAN")
    game.create_window()
    game.start()
