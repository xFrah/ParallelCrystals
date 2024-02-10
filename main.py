# pygame application to display a matrix, to each cell is assigned a number

import pygame
import sys
import random
import time

# at start only one cell is alive, choose a random number
def init_matrix(width, height):
    matrix = [[0 for x in range(width)] for y in range(height)]
    random_y = random.randint(0, 9)
    random_x = random.randint(0, 9)
    random_number = random.randint(1, 9)
    matrix[random_y][random_x] = random_number
    return matrix


def main():
    # screen size
    width = 10
    height = 10
    size = 50
    # init pygame
    pygame.init()
    screen = pygame.display.set_mode((width * size, height * size))
    pygame.display.set_caption("Matrix")
    clock = pygame.time.Clock()
    # init matrix
    matrix = init_matrix(width, height)
    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # draw matrix
        for y in range(height):
            for x in range(width):
                pygame.draw.rect(screen, (255, 255, 255), (x * size, y * size, size, size), 1)
                if matrix[y][x] != 0:
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(matrix[y][x]), 1, (255, 255, 255))
                    screen.blit(text, (x * size + 20, y * size + 20))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
