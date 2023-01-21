import pygame

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0,0,0)
BG_COLOR = (20, 20, 20)

HEAD_COLOR = [(124, 0, 232), (97, 0, 181)]
BODY_COLOR = [(0, 76, 255), (0, 54, 181)]

GOAL_IMG = pygame.image.load("./assets/imgs/goal.png")
BOMB_IMG = pygame.image.load("./assets/imgs/bomb.png")

BLOCK_SIZE = 20

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
N_HIDDEN_LAYERS = 512
GAMMA = 0.9

GAME_SPEED = 100