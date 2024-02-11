import pygame
import sys
import random

# Constants
WIDTH = 800  # in pixels
HEIGHT = 600
num_particles = 100
gray = (200, 200, 200)
black = (0, 0, 0)
dark_red = (128, 0, 0)
yellow = (255, 255, 0)


class Particle:
    def __init__(self, iid):
        self.id = iid
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.walker = random.choice([True, False])
        self.y_index = None


# pygame window that can display particle
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random Walk")

    particles = {i: Particle(i) for i in range(num_particles)}
    particles_x = [i for i in range(num_particles)]
    particles_y = [i for i in range(num_particles)]

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        collisions = set()

        screen.fill(black)

        particles_x.sort(key=lambda i: particles[i].x)
        particles_y.sort(key=lambda i: particles[i].y)

        for i, j in enumerate(particles_y):
            particles[j].y_index = i

        for _, particle in particles.items():
            if particle.walker:
                particle.x += random.randint(-1, 1)
                particle.y += random.randint(-1, 1)

        # loop through particles_x and flag for possible collision if two consecutive particles are too close
        for i in range(num_particles - 1):
            j = i + 1
            while j < num_particles and abs(particles[particles_x[j]].x - particles[particles_x[i]].x) < 5:
                # check if one of the particles is a walker
                if not (particles[particles_x[i]].walker and particles[particles_x[j]].walker) and (particles[particles_x[i]].walker or particles[particles_x[j]].walker):
                    pygame.draw.line(
                        screen,
                        gray,
                        (particles[particles_x[i]].x, particles[particles_x[i]].y),
                        (particles[particles_x[j]].x, particles[particles_x[j]].y),
                        1,
                    )
                # check if the two particles are close in the y direction by while looping starting around the y_index
                k = particles[particles_x[i]].y_index + 1
                while k < num_particles and abs(particles[particles_y[k]].y - particles[particles_x[i]].y) <= 5:
                    # check if both particles are walkers
                    if not (particles[particles_x[i]].walker and particles[particles_y[k]].walker) and (particles[particles_x[i]].walker or particles[particles_x[j]].walker):
                        pygame.draw.line(
                            screen,
                            yellow,
                            (particles[particles_x[i]].x, particles[particles_x[i]].y),
                            (particles[particles_y[k]].x, particles[particles_y[k]].y),
                            2,
                        )
                    # collisions.add((particles_x[i], particles_x[j]))
                    # particles[particles_x[i]].walker = False
                    # particles[particles_y[k]].walker = False
                    k += 1

                k = particles[particles_x[i]].y_index - 1
                while k >= 0 and abs(particles[particles_y[k]].y - particles[particles_x[i]].y) <= 5:
                    if particles[particles_x[i]].id == particles[particles_y[k]].id:
                        print(particles[particles_x[i]].id, particles[particles_y[k]].id)
                        if not (particles[particles_x[i]].walker and particles[particles_y[k]].walker):
                            pygame.draw.line(
                                screen,
                                yellow,
                                (particles[particles_x[i]].x, particles[particles_x[i]].y),
                                (particles[particles_y[k]].x, particles[particles_y[k]].y),
                                2,
                            )
                        # collisions.add((particles_x[i], particles_x[j]))
                        # particles[particles_x[i]].walker = False
                        # particles[particles_y[k]].walker = False
                    k -= 1

                j += 1

        for _, particle in particles.items():
            pygame.draw.circle(
                screen,
                gray if particle.walker else dark_red,
                (particle.x, particle.y),
                3,
                3,
            )

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
