import pygame # This is the engine used in the game
from sys import exit # To close the window when the game is over
from os import environ # To center the game window the best possible
from random import randrange # Random numbers used for the food
import logging # Logging function for movements and errors
logger = logging.getLogger(__name__) # Setting logger
environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window
var = GlobalVariables() # Initializing global variables


class GlobalVariables():
    """Global variables to be used while drawing and moving the snake game.

    Attributes:
        BOARD_SIZE: The size in blocks of the snake game.
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
        self.HEAD_COLOR = (0, 100, 0)
        self.BODY_COLOR = (0, 200, 0)
        self.FOOD_COLOR = (200, 0, 0)
        self.GAME_SPEED = 24

        if self.BOARD_SIZE >= 50:
            logger.warning('WARNING: BOARD IS TOO BIG, IT MAY RUN SLOWER')

class Snake():
    """Player (snake) class which initializes head, body and orientation.

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
        self.orientation = "RIGHT"

    def change_orientation(self, orientation):
        """Change current orientation given a set of rules."""
        if orientation == "RIGHT" and not self.orientation == "LEFT":
            self.orientation = "RIGHT"
        elif orientation == "LEFT" and not self.orientation == "RIGHT":
            self.orientation = "LEFT"
        elif orientation == "UP" and not self.orientation == "DOWN":
            self.orientation = "UP"
        elif orientation == "DOWN" and not self.orientation == "UP":
            self.orientation = "DOWN"

    def move(self, food_pos):
        """According to orientation, move 1 block. If the head is not positioned
        on food, pop a body part. Else (food), return without popping."""
        if self.orientation == "RIGHT":
            self.head[0] += 1
        elif self.orientation == "LEFT":
            self.head[0] -= 1
        elif self.orientation == "UP":
            self.head[1] -= 1
        elif self.orientation == "DOWN":
            self.head[1] += 1

        self.body.insert(0, list(self.head))

        if self.head == food_pos:
            logger.info('EVENT: FOOD EATEN')
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

        for body_part in self.body[1:]:
            if self.head == body_part:
                logger.info('EVENT: BODY COLLISION')
                return True

        return False

    def return_body(self):
        """Return the whole body."""
        return self.body


class FoodGenerator():
    """Generate and keep track of food.

    Attributes:
        pos: Current position of food.
        is_food_on_screen: Flag for existence of food.
    """
    def __init__(self):
        """Initialize a food piece and set existence flag."""
        self.pos = [randrange(1, var.BOARD_SIZE), randrange(1, var.BOARD_SIZE)]
        self.is_food_on_screen = True

    def generate_food(self, body):
        """Generate food and verify if it's on a valid place."""
        if self.is_food_on_screen == False:
            while True:
                self.pos = [randrange(1, var.BOARD_SIZE), randrange(1, \
                                                                var.BOARD_SIZE)]
                misplaced = False # Keeps track of food placed in the body.

                for body_part in body[1:]:
                    if self.pos == body_part:
                        misplaced = True

                if misplaced is True:
                    pass
                else:
                    break

            logger.info('EVENT: FOOD APPEARED')
            self.is_food_on_screen = True

        return self.pos

    def set_food_on_screen(self, bool_value):
        """Set flag for existence (or not) of food."""
        self.is_food_on_screen = bool_value

class Game():
    """Hold the game window and functions.

    Attributes:
        window: pygame window to show the game.
        fps: Define Clock and ticks in which the game will be displayed.
        score: Keeps track of how many food pieces were eaten.
    """
    def __init__(self):
        """Initialize window, fps and score."""
        self.window = pygame.display.set_mode((var.BOARD_SIZE * var.BLOCK_SIZE,\
                                            var.BOARD_SIZE * var.BLOCK_SIZE))
        self.fps = pygame.time.Clock()
        self.score = 0

    def start(self):
        """Create some wait time before the actual drawing of the game."""
        for i in range(3):
            pygame.display.set_caption("SNAKE GAME  |  Game starts in " +\
                                       str(3 - i) + " second(s) ...")
            pygame.time.wait(1000)
        logger.info('EVENT: GAME START')

    def over(self):
        """If collision with wall or body, end the game."""
        pygame.display.set_caption("SNAKE GAME  |  Score: " + str(self.score) +\
            "  |  GAME OVER. Press any SPACE or ESC to quit ...")
        logger.info('EVENT: GAME OVER')

        while True:
            keys = pygame.key.get_pressed()
            pygame.event.pump()

            if keys[pygame.K_ESCAPE] or keys[pygame.K_Q]:
                logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
                break

        pygame.quit()
        exit()

    def handle_input(self, snake, keys):
        """After getting current pressed keys, handle important cases."""
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            logger.info('ACTION: KEY PRESSED: ESCAPE or Q')
            self.over()
        elif keys[pygame.K_RIGHT]:
            logger.info('ACTION: KEY PRESSED: RIGHT')
            snake.change_orientation("RIGHT")
        elif keys[pygame.K_LEFT]:
            logger.info('ACTION: KEY PRESSED: LEFT')
            snake.change_orientation("LEFT")
        elif keys[pygame.K_UP]:
            logger.info('ACTION: KEY PRESSED: UP')
            snake.change_orientation("UP")
        elif keys[pygame.K_DOWN]:
            logger.info('ACTION: KEY PRESSED: DOWN')
            snake.change_orientation("DOWN")

def main():
    """The main function where the game will be executed."""
    # Setup basic configurations for logging in this module
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
    snake = Snake()
    foodgenerator = FoodGenerator()
    game = Game()

    game.start()

    # The main loop, it pump key_presses and update the board every tick.
    while True:
        keys = pygame.key.get_pressed()
        pygame.event.pump()
        game.handle_input(snake, keys)

        food_pos = foodgenerator.generate_food(snake.return_body())
        if snake.move(food_pos) == True:
            game.score += 1
            foodgenerator.set_food_on_screen(False)

        game.window.fill(pygame.Color(225, 225, 225))

        head = 1
        for pos in snake.return_body():
            if head == 1:
                pygame.draw.rect(game.window, var.HEAD_COLOR,\
                            pygame.Rect(pos[0]*var.BLOCK_SIZE, pos[1] *\
                            var.BLOCK_SIZE, var.BLOCK_SIZE, var.BLOCK_SIZE))
                head = 0
            else:
                pygame.draw.rect(game.window, var.BODY_COLOR, \
                            pygame.Rect(pos[0] * var.BLOCK_SIZE, pos[1] *\
                            var.BLOCK_SIZE, var.BLOCK_SIZE, var.BLOCK_SIZE))

        pygame.draw.rect(game.window, var.FOOD_COLOR, pygame.Rect(food_pos[0]\
                         * var.BLOCK_SIZE, food_pos[1] * var.BLOCK_SIZE,\
                         var.BLOCK_SIZE, var.BLOCK_SIZE))

        if snake.check_collision() == True:
            game.over()

        pygame.display.set_caption("SNAKE GAME  |  Score: " + str(game.score))
        pygame.display.update()
        game.fps.tick(var.GAME_SPEED)

if __name__ == '__main__':
    main() # Execute game! Let's play ;)
