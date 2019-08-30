# This source has created by Maickonn Richard.
# https://github.com/Maickonn

import sys
import pygame
import random
import numpy

from pygame.locals import *
from utils.game_objects import Player
from utils.game_objects import Ball
from utils.neural_network import NeuralNetwork


# Constants
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 600
MAX_RANDOM_GENOME = 1

# Initialization
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyPong Neural")
clock = pygame.time.Clock()

# Score system
score = 0
score_record = 0
score_font = pygame.font.SysFont("Arial", 18)

# Controllers
moving_left = False
moving_right = False
move_direction = "standby"

# Games objects
ai_player = Player(150, 20, WINDOW_WIDTH / 2.5, WINDOW_HEIGHT - 20) # Width, height, x position, y position
my_ball = Ball(350, 0) # x position, y position

# Setup neural network
neural_network = NeuralNetwork()
losses_count = 0


while True:
    clock.tick(100) # Lock on 100 FPS

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # A.I directions
    ai_result = neural_network.get_output(numpy.array([[ai_player.rect.x / WINDOW_WIDTH, my_ball.rect.x / WINDOW_WIDTH]])) # Get result of the neural
    if ai_result[0][0] >= 0.6: # This can be any value of 0.0 to 1.0
        move_direction = "right"
    elif ai_result[0][0] <= 0.4:
        move_direction = "left"
    else:
        move_direction = "standby"

    # A.I movements
    if (move_direction == "right") and (ai_player.rect.x < WINDOW_WIDTH - ai_player.width):
        ai_player.rect.x += ai_player.speed
    elif (move_direction == "left") and (ai_player.rect.x > 0):
        ai_player.rect.x -= ai_player.speed

    # Ball movements
    my_ball.rect.x += my_ball.drop_direction
    my_ball.rect.y += my_ball.speed
    if my_ball.rect.x <= 0:
        while my_ball.rect.x <= 0: # This prevent the ball exceed borders
            my_ball.rect.x += 1
        my_ball.drop_direction *= -1
    elif my_ball.rect.x >= WINDOW_WIDTH - my_ball.rect.w:
        while my_ball.rect.x >= WINDOW_WIDTH - my_ball.rect.w:
            my_ball.rect.x -= 1
        my_ball.drop_direction *= -1
    elif my_ball.rect.y <= 0:
        while my_ball.rect.y <= 0:
            my_ball.rect.y += 1
        my_ball.speed *= -1

    # Check collisions
    if my_ball.rect.colliderect(ai_player.rect):
        # This fix the bug on lateral collision
        while (my_ball.rect.y + ai_player.rect.h) >= ai_player.rect.y:
            my_ball.rect.y -= 1

        score += 1
        my_ball.speed *= -1
        if my_ball.drop_direction < 0:
            my_ball.drop_direction = random.randint(-10, -1)
        elif my_ball.drop_direction > 0:
            my_ball.drop_direction = random.randint(1, 10)
        else:
            my_ball.drop_direction = random.randint(-10, 10)

    # Check game over
    if my_ball.rect.y >= WINDOW_HEIGHT:
        if score > score_record:
            score_record = score

        # This make the A.I learn with self error
        for i in range(50):
            if ai_player.rect.x < my_ball.rect.x:
                neural_network.train(ai_result, numpy.array([1]))
            else:
                neural_network.train(ai_result, numpy.array([0]))

            ai_result = neural_network.get_output(numpy.array([[ai_player.rect.x / WINDOW_WIDTH, my_ball.rect.x / WINDOW_WIDTH]]))  # Get result of the neural

        print("New mutation: " + str(neural_network.input_weights) + " - " + str(neural_network.hidden_weights))

        score = 0
        losses_count += 1
        del my_ball
        my_ball = Ball(random.randint(100, 600), 1)


    # Screen updates
    screen.fill((0, 0, 0))
    score_obj = score_font.render("Score: " + str(score) + " | Score Record: " + str(score_record) + " | Losses: " + str(losses_count), True, (255, 255, 255))
    screen.blit(score_obj, score_obj.get_rect())
    screen.blit(ai_player.pygame_object, ai_player.rect)
    screen.blit(my_ball.pygame_object, my_ball.rect)
    pygame.display.update()
