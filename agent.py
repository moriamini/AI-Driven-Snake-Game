import pygame
import torch
import random
import numpy as np
from collections import deque
from src.game import SnakeGameAI, Direction
from src.libs.game_object import GameObject, Point
from src.model import LinearQNet, QTrainer
import os
import pandas as pd
from src.config import MAX_MEMORY, BATCH_SIZE, LR, N_HIDDEN_LAYERS, GAMMA, GAME_SPEED

class Agent:
    def __init__(self, last_model_address = None):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self. model = LinearQNet(14, N_HIDDEN_LAYERS, 3)
        
        if last_model_address is not None and os.path.exists(last_model_address):
            self.model.load_state_dict(torch.load(last_model_address))
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, game):
        snake_head = game.snake_body[0]
        point_left = GameObject(snake_head.position.x - 20, snake_head.position.y, 'snake_left')
        point_right = GameObject(snake_head.position.x + 20, snake_head.position.y, 'snake_right')
        point_up = GameObject(snake_head.position.x, snake_head.position.y - 20, 'snake_up')
        point_down = GameObject(snake_head.position.x, snake_head.position.y + 20, 'snake_down')
        
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),
            
            # Danger right
            (direction_right and game.is_collision(point_down)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)),
            
            # Danger left
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_down and game.is_collision(point_right)),

            # Danger bomb straight
            (direction_right and point_right == game.bomb_obj.position) or
            (direction_left and point_left == game.bomb_obj.position) or
            (direction_up and point_up == game.bomb_obj.position) or
            (direction_down and point_down == game.bomb_obj.position),
            
            # Danger bomb right
            (direction_right and point_down == game.bomb_obj.position) or
            (direction_left and point_up == game.bomb_obj.position) or
            (direction_up and point_right == game.bomb_obj.position) or
            (direction_down and point_left == game.bomb_obj.position),
            
            # Danger bomb left
            (direction_right and point_up == game.bomb_obj.position) or
            (direction_left and point_down == game.bomb_obj.position) or
            (direction_up and point_left == game.bomb_obj.position) or
            (direction_down and point_right == game.bomb_obj.position),
            
            # Move Direction,
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Food location
            game.food_obj.position.x < game.snake_head.position.x, # food left
            game.food_obj.position.x > game.snake_head.position.x, # food rigt
            game.food_obj.position.y < game.snake_head.position.y, # food up
            game.food_obj.position.y > game.snake_head.position.y, # food down
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
    
def start():
    score_df = []
    total_scores = 0
    record = 0
    agent = Agent('./export/model.pt')
    game = SnakeGameAI(speed=GAME_SPEED)
    
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move action
        final_move = agent.get_action(state_old)
        
        # preform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory and save model
            agent.n_games += 1
            
            agent.train_long_memory()
            agent.model.save()
            
            if score > record:
                record = score
                
            total_scores += score
            mean_score = total_scores / agent.n_games
            
            score_df.append({'score': score, 'mean_score': mean_score, 'round': (agent.n_games + 1), 'total_score': total_scores, 'record': record})
            
            pd.DataFrame(score_df).to_csv('./export/result.csv')
            
            game.reset(round_of_game=(agent.n_games + 1), high_record=record)

if __name__ == '__main__':
    start()