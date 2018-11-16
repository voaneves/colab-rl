#!/usr/bin/env python

"""SnakeGame: A simple and fun exploration, meant to be used by Human and AI.
"""

import sys  # To close the window when the game is over
from array import array  # Efficient numeric arrays
from os import environ, path  # To center the game window the best possible
import random  # Random numbers used for the food
import logging  # Logging function for movements and errors
from itertools import tee  # For the color gradient on snake
import pygame  # This is the engine used in the game
import numpy as np

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

# Actions, options and forbidden moves
options = {'QUIT': 0, 'PLAY': 1, 'BENCHMARK': 2, 'LEADERBOARDS': 3, 'MENU': 4,
           'ADD_LEADERBOARDS': 5}
relative_actions = {'LEFT': 0, 'FORWARD': 1, 'RIGHT': 2}
actions = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'IDLE': 4}
forbidden_moves = [(0, 1), (1, 0), (2, 3), (3, 2)]

# Possible rewards in the game
rewards = {'MOVE': -0.005, 'GAME_OVER': -1, 'SCORED': 1}

# Types of point in the board
point_type = {'EMPTY': 0, 'FOOD': 1, 'BODY': 2, 'HEAD': 3, 'DANGEROUS': 4}

# Speed levels possible to human players, MEGA HARDCORE starts with MEDIUM and
# increases with snake size
levels = [" EASY ", " MEDIUM ", " HARD ", " MEGA HARDCORE "]
speeds = {'EASY': 80, 'MEDIUM': 60, 'HARD': 40, 'MEGA_HARDCORE': 65}


class GlobalVariables:
    """Global variables to be used while drawing and moving the snake game.

    Attributes
    ----------
    BOARD_SIZE: int, optional, default = 30
        The size of the board.
    BLOCK_SIZE: int, optional, default = 20
        The size in pixels of a block.
    HEAD_COLOR: tuple of 3 * int, optional, default = (42, 42, 42)
        Color of the head. Start of the body color gradient.
    TAIL_COLOR: tuple of 3 * int, optional, default = (152, 152, 152)
        Color of the tail. End of the body color gradient.
    FOOD_COLOR: tuple of 3 * int, optional, default = (200, 0, 0)
        Color of the food.
    GAME_SPEED: int, optional, default = 10
        Speed in ticks of the game. The higher the faster.
    BENCHMARK: int, optional, default = 10
        Ammount of matches to BENCHMARK and possibly go to leaderboards.
    """
    def __init__(self, BOARD_SIZE = 30, BLOCK_SIZE = 20,
                 HEAD_COLOR = (42, 42, 42), TAIL_COLOR = (152, 152, 152),
                 FOOD_COLOR = (200, 0, 0), GAME_SPEED = 80, GAME_FPS = 100,
                 BENCHMARK = 10):
        """Initialize all global variables. Can be updated with argument_handler.
        """
        self.BOARD_SIZE = BOARD_SIZE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.HEAD_COLOR = HEAD_COLOR
        self.TAIL_COLOR = TAIL_COLOR
        self.FOOD_COLOR = FOOD_COLOR
        self.GAME_SPEED = GAME_SPEED
        self.GAME_FPS = GAME_FPS
        self.BENCHMARK = BENCHMARK

        if self.BOARD_SIZE > 50: # Warn the user about performance
            logger.warning('WARNING: BOARD IS TOO BIG, IT MAY RUN SLOWER.')

class TextBlock:
    """Block of text class, used by pygame. Can be used to both text and menu.

    Attributes:
    ----------
    text: string
        The text to be displayed.
    pos: tuple of 2 * int
        Color of the tail. End of the body color gradient.
    screen: pygame window object
        The screen where the text is drawn.
    scale: int, optional, default = 1 / 12
        Adaptive scale to resize if the board size changes.
    type: string, optional, default = "text"
        Assert whether the BlockText is a text or menu option.
    """
    def __init__(self, text, pos, screen, scale = (1 / 12), type = "text"):
        """Initialize, set position of the rectangle and render the text block."""
        self.type = type
        self.hovered = False
        self.text = text
        self.pos = pos
        self.screen = screen
        self.scale = scale
        self.set_rect()
        self.draw()

    def draw(self):
        """Set what to render and blit on the pygame screen."""
        self.set_rend()
        self.screen.blit(self.rend, self.rect)

    def set_rend(self):
        """Set what to render (font, colors, sizes)"""
        font = pygame.font.Font(resource_path("resources/fonts/freesansbold.ttf"),
                                int((var.BOARD_SIZE * var.BLOCK_SIZE) * self.scale))
        self.rend = font.render(self.text, True, self.get_color(),
                                self.get_background())

    def get_color(self):
        """Get color to render for text and menu (hovered or not).

        Return
        ----------
        color: tuple of 3 * int
            The color that will be rendered for the text block.
        """
        color = pygame.Color(42, 42, 42)

        if self.type == "menu":
            if self.hovered:
                pass
            else:
                color = pygame.Color(152, 152, 152)

        return color

    def get_background(self):
        """Get background color to render for text (hovered or not) and menu.

        Return
        ----------
        color: tuple of 3 * int
            The color that will be rendered for the background of the text block.
        """
        color = None

        if self.type == "menu":
            if self.hovered:
                color = pygame.Color(152, 152, 152)

        return color

    def set_rect(self):
        """Set the rectangle and it's position to draw on the screen."""
        self.set_rend()
        self.rect = self.rend.get_rect()
        self.rect.center = self.pos


class Snake:
    """Player (snake) class which initializes head, body and board.

    The body attribute represents a list of positions of the body, which are in-
    cremented when moving/eating on the position [0]. The orientation represents
    where the snake is looking at (head) and collisions happen when any element
    is superposed with the head.

    Attributes
    ----------
    head: list of 2 * int, default = [BOARD_SIZE / 4, BOARD_SIZE / 4]
        The head of the snake, located according to the board size.
    body: list of lists of 2 * int
        Starts with 3 parts and grows when food is eaten.
    previous_action: int, default = 1
        Last action which the snake took.
    length: int, default = 3
        Variable length of the snake, can increase when food is eaten.
    """
    def __init__(self):
        """Inits Snake with 3 body parts (one is the head) and pointing right"""
        self.head = [int(var.BOARD_SIZE / 4), int(var.BOARD_SIZE / 4)]
        self.body = [[self.head[0], self.head[1]],
                     [self.head[0] - 1, self.head[1]],
                     [self.head[0] - 2, self.head[1]]]
        self.previous_action = 1
        self.length = 3

    def is_movement_invalid(self, action):
        valid = False

        if (action, self.previous_action) in forbidden_moves:
            valid = True

        return valid

    def move(self, action, food_pos):
        """According to orientation, move 1 block. If the head is not positioned
        on food, pop a body part. Else, return without popping.

        Return
        ----------
        ate_food: boolean
            Flag which represents whether the snake ate or not food.
        """
        ate_food = False

        if action == actions['IDLE'] or self.is_movement_invalid(action):
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

            ate_food = True
        else:
            self.body.pop()

        return ate_food


class FoodGenerator:
    """Generate and keep track of food.

    Attributes
    ----------
    pos:
        Current position of food.
    is_food_on_screen:
        Flag for existence of food.
    """
    def __init__(self, body):
        """Initialize a food piece and set existence flag."""
        self.is_food_on_screen = False
        self.pos = self.generate_food(body)

    def generate_food(self, body):
        """Generate food and verify if it's on a valid place.

        Return
        ----------
        pos: tuple of 2 * int
            Position of the food that was generated. It can't be in the body.
        """
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


class Game:
    """Hold the game window and functions.

    Attributes
    ----------
    window: pygame display
        Pygame window to show the game.
    fps: pygame time clock
        Define Clock and ticks in which the game will be displayed.
    snake: object
        The actual snake who is going to be played.
    food_generator: object
        Generator of food which responds to the snake.
    food_pos: tuple of 2 * int
        Position of the food on the board.
    game_over: boolean
        Flag for game_over.
    player: string
        Define if human or robots are playing the game.
    board_size: int, optional, default = 30
        The size of the board.
    local_state: boolean, optional, default = False
        Whether to use or not game expertise (used mostly by robots players).
    relative_pos: boolean, optional, default = False
        Whether to use or not relative position of the snake head. Instead of
        actions, use relative_actions.
    screen_rect: tuple of 2 * int
        The screen rectangle, used to draw relatively positioned blocks.
    """
    def __init__(self, player, board_size = 30, local_state = False, relative_pos = False):
        """Initialize window, fps and score. Change nb_actions if relative_pos"""
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
        """Reset the game environment."""
        self.step = 0
        self.snake = Snake()
        self.food_generator = FoodGenerator(self.snake.body)
        self.food_pos = self.food_generator.pos
        self.scored = False
        self.game_over = False

    def create_window(self):
        """Create a pygame display with BOARD_SIZE * BLOCK_SIZE dimension."""
        pygame.init()

        flags = pygame.DOUBLEBUF
        self.window = pygame.display.set_mode((var.BOARD_SIZE * var.BLOCK_SIZE,\
                                               var.BOARD_SIZE * var.BLOCK_SIZE),
                                               flags)
        self.window.set_alpha(None)
        self.screen_rect = self.window.get_rect()
        self.fps = pygame.time.Clock()

    def cycle_menu(self, menu_options, list_menu, dict, img = None,
                   img_rect = None):
        """"""
        selected = False
        selected_option = None

        while not selected:
            pygame.event.pump()
            events = pygame.event.get()

            self.window.fill(pygame.Color(225, 225, 225))

            for i, option in enumerate(menu_options):
                if option is not None:
                    option.draw()
                    option.hovered = False

                    if option.rect.collidepoint(pygame.mouse.get_pos()):
                        option.hovered = True

                        for event in events:
                            if event.type == pygame.MOUSEBUTTONUP:
                                selected_option = dict[list_menu[i]]

            if selected_option is not None:
                selected = True
            if img is not None:
                self.window.blit(img, img_rect.bottomleft)

            pygame.display.update()

        return selected_option

    def cycle_matches(self, n_matches = 10, mega_hardcore = False):
        """"""
        self.reset_game()
        score = array('i')

        for match in range(n_matches):
            score.append(self.single_player(mega_hardcore))

        return score

    def menu(self):
        """Main menu of the game.

        Return
        ----------
        selected_option: int
            The selected option in the main loop.
        """
        pygame.display.set_caption("SNAKE GAME  | PLAY NOW!")

        img = pygame.image.load(resource_path("resources/images" +
                                              "/snake_logo.png")).convert()
        img = pygame.transform.scale(img, (var.BOARD_SIZE * var.BLOCK_SIZE,
                                     int(var.BOARD_SIZE * var.BLOCK_SIZE / 3)))
        img_rect = img.get_rect()
        img_rect.center = self.screen_rect.center
        list_menu = ['PLAY', 'BENCHMARK', 'LEADERBOARDS', 'QUIT']
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
        selected_option = self.cycle_menu(menu_options, list_menu, options,
                                          img, img_rect)

        return selected_option

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
                var.GAME_SPEED, mega_hardcore = self.select_speed()
                score = self.cycle_matches(n_matches = 1,
                                           mega_hardcore = mega_hardcore)
                opt = self.over(score)
            elif opt == options['BENCHMARK']:
                var.GAME_SPEED, mega_hardcore = self.select_speed()
                score = self.cycle_matches(n_matches = var.BENCHMARK,
                                           mega_hardcore = mega_hardcore)
                opt = self.over(score)
            elif opt == options['LEADERBOARDS']:
                pass
            elif opt == options['ADD_LEADERBOARDS']:
                pass
            elif opt == options['MENU']:
                opt = self.menu()

    def over(self, score):
        """If collision with wall or body, end the game and open options.

        Return
        ----------
        selected_option: int
            The selected option in the main loop.
        """
        score_option = None

        if len(score) == var.BENCHMARK:
            score_option = TextBlock(' ADD TO LEADERBOARDS ',
                                        (self.screen_rect.centerx,
                                         8 * self.screen_rect.centery / 10),
                                        self.window, (1 / 15), "menu")

        text_score = 'SCORE: ' + str(np.mean(score))
        list_menu = ['PLAY', 'MENU', 'ADD_LEADERBOARDS', 'QUIT']
        menu_options = [TextBlock(' PLAY AGAIN ', (self.screen_rect.centerx,
                                            4 * self.screen_rect.centery / 10),
                                            self.window, (1 / 15), "menu"),
                           TextBlock(' GO TO MENU ', (self.screen_rect.centerx,
                                            6 * self.screen_rect.centery / 10),
                                            self.window, (1 / 15), "menu"),
                           score_option,
                           TextBlock(' QUIT ', (self.screen_rect.centerx,
                                            10 * self.screen_rect.centery / 10),
                                            self.window, (1 / 15), "menu"),
                           TextBlock(text_score, (self.screen_rect.centerx,
                                             15 * self.screen_rect.centery / 10),
                                             self.window, (1 / 10), "text")]
        pygame.display.set_caption("SNAKE GAME  | " + text_score
                                   + "  |  GAME OVER...")
        logger.info('EVENT: GAME OVER | FINAL ' + text_score)
        selected_option = self.cycle_menu(menu_options, list_menu, options)

        return selected_option

    def select_speed(self):
        """Speed menu, right before calling start_match.

        Return
        ----------
        speed: int
            The selected speed in the main loop.
        """
        list_menu = ['EASY', 'MEDIUM', 'HARD', 'MEGA_HARDCORE']
        menu_options = [TextBlock(levels[0], (self.screen_rect.centerx,
                                              4 * self.screen_rect.centery / 10),
                                              self.window, (1 / 10), "menu"),
                        TextBlock(levels[1], (self.screen_rect.centerx,
                                              8 * self.screen_rect.centery / 10),
                                              self.window, (1 / 10), "menu"),
                        TextBlock(levels[2], (self.screen_rect.centerx,
                                              12 * self.screen_rect.centery / 10),
                                              self.window, (1 / 10), "menu"),
                        TextBlock(levels[3], (self.screen_rect.centerx,
                                              16 * self.screen_rect.centery / 10),
                                              self.window, (1 / 10), "menu")]

        speed = self.cycle_menu(menu_options, list_menu, speeds)
        mega_hardcore = False

        if speed == speeds['MEGA_HARDCORE']:
            mega_hardcore = True

        return speed, mega_hardcore

    def single_player(self, mega_hardcore = False):
        """Game loop for single_player (HUMANS).

        Return
        ----------
        score: int
            The final score for the match (discounted of initial length).
        """
        # The main loop, it pump key_presses and update the board every tick.
        previous_size = self.snake.length # Initial size of the snake
        current_size = previous_size # Initial size
        color_list = self.gradient([(42, 42, 42), (152, 152, 152)],\
                                   previous_size)

        # Main loop, where snakes moves after elapsed time is bigger than the
        # move_wait time. The last_key pressed is recorded to make the game more
        # smooth for human players.
        elapsed = 0
        last_key = self.snake.previous_action
        move_wait = var.GAME_SPEED

        while not self.game_over:
            elapsed += self.fps.get_time()  # Get elapsed time since last call.

            if mega_hardcore:  # Progressive speed increments, the hardest.
                move_wait = var.GAME_SPEED - (2 * (self.snake.length - 3))

            key_input = self.handle_input()  # Receive inputs with tick.
            invalid_key = self.snake.is_movement_invalid(key_input)

            if key_input is not None and not invalid_key:
                last_key = key_input

            if elapsed >= move_wait:  # Move and redraw
                elapsed = 0
                self.game_over = self.play(last_key)
                current_size = self.snake.length  # Update the body size

                if current_size > previous_size:
                    color_list = self.gradient([(42, 42, 42), (152, 152, 152)],
                                                   current_size)

                    previous_size = current_size

                self.draw(color_list)

            pygame.display.update()
            self.fps.tick(100)  # Limit FPS to 100

        score = current_size - 3  # After the game is over, record score

        return score

    def check_collision(self):
        """Check wether any collisions happened with the wall or body.

        Return
        ----------
        collided: boolean
            Whether the snake collided or not.
        """
        collided = False

        if self.snake.head[0] > (var.BOARD_SIZE - 1) or self.snake.head[0] < 0:
            logger.info('EVENT: WALL COLLISION')
            collided = True
        elif self.snake.head[1] > (var.BOARD_SIZE - 1) or self.snake.head[1] < 0:
            logger.info('EVENT: WALL COLLISION')
            collided = True
        elif self.snake.head in self.snake.body[1:]:
            logger.info('EVENT: BODY COLLISION')
            collided = True

        return collided

    def is_won(self):
        """Verify if the score is greater than 0.

        Return
        ----------
        won: boolean
            Whether the score is greater than 0.
        """
        return self.snake.length > 3

    def generate_food(self):
        """Generate new food if needed.

        Return
        ----------
        food_pos: tuple of 2 * int
            Current position of the food.
        """
        food_pos = self.food_generator.generate_food(self.snake.body)

        return food_pos

    def handle_input(self):
        """After getting current pressed keys, handle important cases.

        Return
        ----------
        action: int
            Handle human input to assess the next action.
        """
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        keys = pygame.key.get_pressed()
        pygame.event.pump()
        action = None

        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
            self.over(self.snake.length - 3)
        elif keys[pygame.K_LEFT]:
            logger.info('ACTION: KEY PRESSED: LEFT')
            action = actions['LEFT']
        elif keys[pygame.K_RIGHT]:
            logger.info('ACTION: KEY PRESSED: RIGHT')
            action = actions['RIGHT']
        elif keys[pygame.K_UP]:
            logger.info('ACTION: KEY PRESSED: UP')
            action = actions['UP']
        elif keys[pygame.K_DOWN]:
            logger.info('ACTION: KEY PRESSED: DOWN')
            action = actions['DOWN']

        return action

    def eval_local_safety(self, canvas, body):
        """Evaluate the safety of the head's possible next movements.

        Return
        ----------
        canvas: np.array of size BOARD_SIZE**2
            After using game expertise, change canvas values to DANGEROUS if true.
        """
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
        """Create a matrix of the current state of the game.

        Return
        ----------
        canvas: np.array of size BOARD_SIZE**2
            Return the current state of the game in a matrix.
        """
        canvas = np.zeros((var.BOARD_SIZE, var.BOARD_SIZE))

        if self.game_over:
            pass
        else:
            body = self.snake.body

            for part in body:
                canvas[part[0], part[1]] = point_type['BODY']

            canvas[body[0][0], body[0][1]] = point_type['HEAD']

            if self.local_state:
                canvas = self.eval_local_safety(canvas, body)

            canvas[self.food_pos[0], self.food_pos[1]] = point_type['FOOD']

        return canvas

    def relative_to_absolute(self, action):
        """Translate relative actions to absolute.

        Return
        ----------
        action: int
            Translated action from relative to absolute.
        """
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
            self.food_generator.is_food_on_screen = False

        if self.player == "HUMAN":
            if self.check_collision():
                return True
        elif self.check_collision() or self.step > 50 * self.snake.length:
            self.game_over = True

    def get_reward(self):
        """Return the current score. Can be used as the reward function.

        Return
        ----------
        reward: float
            Current reward of the game.
        """
        reward = rewards['MOVE']

        if self.game_over:
            reward = rewards['GAME_OVER']
        elif self.scored:
            reward = self.snake.length

        return reward

    def gradient(self, colors, steps, components = 3):
        """Function to create RGB gradients given 2 colors and steps. If
        component is changed to 4, it does the same to RGBA colors.

        Return
        ----------
        result: list of steps length of tuple of 3 * int (if RGBA, 4 * int)
            List of colors of calculated gradient from start to end.
        """
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

def resource_path(relative_path):
    """Function to return absolute paths. Used while creating .exe file."""
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
