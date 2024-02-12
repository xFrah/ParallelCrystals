import pygame
import sys
import random

# Constants
WIDTH = 800  # in pixels
HEIGHT = 600
num_particles = 200
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

        screen.fill(black)

        particles_x.sort(key=lambda i: particles[i].x)
        particles_y.sort(key=lambda i: particles[i].y)

        for i, j in enumerate(particles_y):
            particles[j].y_index = i

        for _, particle in particles.items():
            if particle.walker:
                particle.x += random.randint(-1, 1)
                particle.y += random.randint(-1, 1)

        checks_for_particle = []

        # loop through particles_x and flag for possible collision if two consecutive particles are too close
        for i in range(num_particles - 1):
            j = i + 1
            check_for_particle = 1

            while j < num_particles and abs(particles[particles_x[j]].x - particles[particles_x[i]].x) <= 7:  # first check
                particle1 = particles[particles_x[i]]
                particle2 = particles[particles_x[j]]
                found = False
                check_for_particle += 1  # check for next iteration, even for stopping the while loop
                j += 1
                if (not particle1.walker and not particle2.walker) or (particle1.walker and particle2.walker):
                    continue

                pygame.draw.line(
                    screen,
                    gray,
                    (particle1.x, particle1.y),
                    (particle2.x, particle2.y),
                    1,
                )

                # check if the two particles are close in the y direction by while looping starting around the y_index
                check_for_particle += 1
                k = particle1.y_index + 1
                while not found and k < num_particles and abs(particles[particles_y[k]].y - particle1.y) <= 7:
                    particle_k = particles[particles_y[k]]
                    k += 1
                    check_for_particle += 1
                    if (not particle1.walker and not particle_k.walker) or (particle1.walker and particle_k.walker):
                        continue
                    # check if both particles are walkers
                    pygame.draw.line(
                        screen,
                        yellow,
                        (particle1.x, particle1.y),
                        (particle_k.x, particle_k.y),
                        2,
                    )
                    if particle2.id == particle_k.id:
                        found = True
                        particle1.walker = False
                        particle2.walker = False

                k = particle1.y_index - 1
                check_for_particle += 1  # first check
                while not found and k >= 0 and abs(particles[particles_y[k]].y - particle1.y) <= 7:
                    particle_k = particles[particles_y[k]]
                    k -= 1
                    check_for_particle += 1  # next iteration check
                    if (not particle1.walker and not particle_k.walker) or (particle1.walker and particle_k.walker):
                        continue
                    pygame.draw.line(
                        screen,
                        yellow,
                        (particle1.x, particle1.y),
                        (particle_k.x, particle_k.y),
                        2,
                    )
                    if particle2.id == particle_k.id:
                        found = True
                        particle1.walker = False
                        particle2.walker = False
            checks_for_particle.append(check_for_particle)

        for _, particle in particles.items():
            pygame.draw.circle(
                screen,
                gray if particle.walker else dark_red,
                (particle.x, particle.y),
                5,
                5,
            )

        t = num_particles**2
        print(f"Avg CPP: {sum(checks_for_particle) / (len(checks_for_particle) + 0.00000001):.02f}")

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
