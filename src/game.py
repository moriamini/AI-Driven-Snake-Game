import pygame
import random
from collections import namedtuple
import numpy as np
from  .config import BG_COLOR, BLACK_COLOR, BLOCK_SIZE, BODY_COLOR, BOMB_IMG, GOAL_IMG, HEAD_COLOR, WHITE_COLOR
from .libs.direction import Direction
from .libs.game_object import GameObject

class SnakeGameAI:
    
    def __init__(self, w=960, h=720, speed = 20, round_of_game: int = 1, high_record: int = 0):
        pygame.init()
        self.font = pygame.font.Font('assets/fonts/oswald.ttf', 18)
        
        self.high_record = high_record
        self.round_of_game = round_of_game
        self.speed = speed; self.width = w; self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('AI Driven Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self, round_of_game=None, high_record=None, speed=None):
        if round_of_game is not None:
            self.round_of_game = round_of_game
        
        if high_record is not None:
            self.high_record = high_record
            
        if speed is not None:
            self.speed = speed
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.snake_head = GameObject(self.width/2, self.height/2, 'snake_head')
        
        self.snake_body = [
            self.snake_head, 
            GameObject(self.snake_head.position.x-BLOCK_SIZE, self.snake_head.position.y, 'snake_body'),
            GameObject(self.snake_head.position.x-(2*BLOCK_SIZE), self.snake_head.position.y, 'snake_body')
        ]
        
        self.score = 0
        self.food_obj = None
        
        # place new food and new bomb
        self._place_food()
        self._place_bomb()
        
        self.frame_iteration = 0
        
    def _place_food(self):
        x_pos = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y_pos = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food_obj = GameObject(x_pos, y_pos, 'food')
        if self.food_obj.position in [snake_point.position for snake_point in self.snake_body]:
            self._place_food()

    def _place_bomb(self):
        x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.bomb_obj = GameObject(x, y, 'bomb')
        if self.bomb_obj.position in [snake_point.position for snake_point in self.snake_body] or self.bomb_obj.position == self.food_obj.position:
            self._place_bomb()
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                quit()
        
        # move
        self._move(action)
        
        # check if game over
        reward = 0; game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake_body) or self.snake_head.position == self.bomb_obj.position:
            game_over = True; reward -= 10
            return reward, game_over, self.score

        self.snake_body.insert(0, self.snake_head)

        # place new food and new bomb or just move
        if self.snake_head.position == self.food_obj.position:
            self.score += 1
            reward += 10
            self._place_food()
            self._place_bomb()
        else:
            self.snake_body.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)
        
        # return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake_head
        
        # hits boundary
        if point.position.x > self.width - BLOCK_SIZE or point.position.x < 0 or point.position.y > self.height - BLOCK_SIZE or point.position.y < 0:
            return True
        
        # hits itself
        if point.position in [snake_point.position for snake_point in self.snake_body[1:]]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BG_COLOR)
        
        pygame.Surface.blit(self.display, GOAL_IMG, (self.food_obj.position.x, self.food_obj.position.y))
        pygame.Surface.blit(self.display, BOMB_IMG, (self.bomb_obj.position.x, self.bomb_obj.position.y))

        pt_first = self.snake_body[0]
        pygame.draw.rect(self.display, HEAD_COLOR[0], pygame.Rect(pt_first.position.x, pt_first.position.y, BLOCK_SIZE, BLOCK_SIZE))
            
        for pt in self.snake_body[1:]:
            pygame.draw.rect(self.display, BODY_COLOR[0], pygame.Rect(pt.position.x, pt.position.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("round: " + str(self.round_of_game) + " , best: " + str(self.high_record) + " , score: " + str(self.score), True, WHITE_COLOR)
        self.display.blit(text, [5, 5])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx]
            
        if np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]
        
        if np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]
            
        self.direction = new_direction
        
        x = self.snake_head.position.x
        y = self.snake_head.position.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.snake_head = GameObject(x, y)