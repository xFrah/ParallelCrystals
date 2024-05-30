import json
import random
import sys
import threading
import time
from typing import List, Tuple, Union

import pygame


def load_config():
    with open("config.json", "r") as file:
        config = json.load(file)

    width = config["screen_width"]
    height = config["screen_height"]
    particle_radius = config["particle_radius"]
    port = config["network_port"]
    num_particles = config["num_particles"]
    cell_size = config["cell_size"]
    return width, height, particle_radius, port, num_particles, cell_size


# Constants
WIDTH, HEIGHT, PARTICLE_RADIUS, PORT, NUM_PARTICLES, CELL_SIZE = load_config()
gray = (200, 200, 200)
black = (0, 0, 0)
dark_red = (128, 0, 0)
yellow = (255, 255, 0)
blue = (0, 0, 255)


class Particle:
    def __init__(self, iid):
        self.iid = iid
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.walker = random.choice([True, False])

    def move(self):
        if self.walker:
            self.x += random.choice([-1, 1])
            self.y += random.choice([-1, 1])

    def __repr__(self):
        return f"Particle({self.iid}, {self.x}, {self.y}, {self.walker})"


def make_lists():
    for particle in particles:
        grid_x = particle.x // CELL_SIZE
        grid_y = particle.y // CELL_SIZE
        if grid_x >= gridWidth:
            grid_x = gridWidth - 1
        if grid_y >= gridHeight:
            grid_y = gridHeight - 1
        if grid_x < 0:
            grid_x = 0
        if grid_y < 0:
            grid_y = 0
        grid[grid_y][grid_x].append(particle)


def reset_linked_lists():
    for row in grid:
        for cell in row:
            cell.clear()


def move_particles():
    for particle in particles:
        particle.move()


def sort_lists():
    for row in grid:
        for cell in row:
            cell.sort(key=lambda p: p.x)


def check_collisions(cell1: List[Particle], cell2: List[Particle], cell2_position: Tuple[int, int]):
    global current_collision, current_checked_cell, current_particle, current_threshold
    current_checked_cell = cell2_position
    time.sleep(0.5)
    i, j = 0, 0
    while i < len(cell1) and j < len(cell2):
        p1 = cell1[i]
        current_particle = p1

        while j < len(cell2) and cell2[j].x < p1.x - PARTICLE_RADIUS:
            j += 1

        k = j
        if k < len(cell2) and not (cell2[k].x <= p1.x + PARTICLE_RADIUS):
            current_collision = (p1, cell2[k])
            current_threshold = cell2[j].x
            time.sleep(0.5)
        while k < len(cell2) and cell2[k].x <= p1.x + PARTICLE_RADIUS:
            current_threshold = cell2[j].x
            p2 = cell2[k]
            current_collision = (p1, p2)
            if p1.walker != p2.walker:
                time.sleep(0.5)
                if abs(p1.y - p2.y) <= PARTICLE_RADIUS:
                    p1.walker = False
                    p2.walker = False
            k += 1
        if k < len(cell2) and not (cell2[k].x <= p1.x + PARTICLE_RADIUS):
            current_collision = (p1, cell2[k])
            current_threshold = cell2[j].x
            time.sleep(0.5)

        i += 1
    current_collision = None
    current_checked_cell = None
    current_threshold = None
    current_particle = None


def check_for_collisions():
    global current_cell
    for row in range(gridHeight):
        for col in range(gridWidth):
            cell = grid[row][col]
            current_cell = (row, col)
            if row == 0 and col == 0:
                # Top left corner
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, grid[row + 1][col + 1], (row + 1, col + 1))
                check_collisions(cell, cell, (row, col))
            elif row == 0 and col == gridWidth - 1:
                # Top right corner
                check_collisions(cell, grid[row + 1][col - 1], (row + 1, col - 1))
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, cell, (row, col))
            elif row == gridHeight - 1 and col == 0:
                # Bottom left corner
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, cell, (row, col))
            elif row == gridHeight - 1 and col == gridWidth - 1:
                # Bottom right corner
                check_collisions(cell, cell, (row, col))
            elif row == 0:
                # Top edge
                check_collisions(cell, grid[row + 1][col - 1], (row + 1, col - 1))
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, grid[row + 1][col + 1], (row + 1, col + 1))
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, cell, (row, col))
            elif row == gridHeight - 1:
                # Bottom edge
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, cell, (row, col))
            elif col == 0:
                # Left edge
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, grid[row + 1][col + 1], (row + 1, col + 1))
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, cell, (row, col))
            elif col == gridWidth - 1:
                # Right edge
                check_collisions(cell, grid[row + 1][col - 1], (row + 1, col - 1))
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, cell, (row, col))
            else:
                # Middle
                check_collisions(cell, grid[row + 1][col - 1], (row + 1, col - 1))
                check_collisions(cell, grid[row + 1][col], (row + 1, col))
                check_collisions(cell, grid[row + 1][col + 1], (row + 1, col + 1))
                check_collisions(cell, grid[row][col + 1], (row, col + 1))
                check_collisions(cell, cell, (row, col))

    current_cell = None


def loop():
    while True:
        make_lists()
        sort_lists()
        check_for_collisions()
        move_particles()
        time.sleep(0.01)
        reset_linked_lists()


if __name__ == "__main__":
    particles = [Particle(i) for i in range(NUM_PARTICLES)]
    gridHeight = HEIGHT // CELL_SIZE
    gridWidth = WIDTH // CELL_SIZE
    grid = [[[] for _ in range(gridWidth)] for _ in range(gridHeight)]

    current_collision: Union[Tuple[Particle, Particle], None] = None
    current_checked_cell: Union[Tuple[int, int], None] = None
    current_cell: Union[Tuple[int, int], None] = None

    threading.Thread(target=loop, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Parallel Crystals Demo")

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(black)

        for i in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(screen, gray, (i, 0), (i, HEIGHT))
        for i in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, gray, (0, i), (WIDTH, i))

        for particle in particles:
            pygame.draw.circle(screen, gray if particle.walker else dark_red, (particle.x, particle.y), PARTICLE_RADIUS)

        if current_cell:
            pygame.draw.rect(screen, blue, (current_cell[1] * CELL_SIZE, current_cell[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

        if current_checked_cell:
            pygame.draw.rect(screen, yellow, (current_checked_cell[1] * CELL_SIZE, current_checked_cell[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

        if current_collision:
            pygame.draw.line(screen, yellow, (current_collision[0].x, current_collision[0].y), (current_collision[1].x, current_collision[1].y), 2)

        # if current_threshold:
        #     # draw vertical line at x = current_threshold and spanning 50 pixels
        #     pygame.draw.line(screen, dark_red, (current_threshold, 0), (current_threshold, HEIGHT), 2)

        if current_particle:
            # draw circle around current_particle
            pygame.draw.circle(screen, yellow, (current_particle.x, current_particle.y), PARTICLE_RADIUS + 2, 2)

        pygame.display.update()
        clock.tick(60)
