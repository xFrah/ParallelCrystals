import random

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Particle(x={self.x}, y={self.y})"

particles = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles2 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles3 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles4 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles5 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles6 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles7 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles8 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]
particles9 = [Particle(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]

# Sort all of them by x
particles.sort(key=lambda p: p.x)
particles2.sort(key=lambda p: p.x)
particles3.sort(key=lambda p: p.x)
particles4.sort(key=lambda p: p.x)
particles5.sort(key=lambda p: p.x)
particles6.sort(key=lambda p: p.x)
particles7.sort(key=lambda p: p.x)
particles8.sort(key=lambda p: p.x)
particles9.sort(key=lambda p: p.x)

adjacent = [particles2, particles3, particles4, particles5, particles6, particles7, particles8, particles9]

def check_collisions(particles1, particles2, threshold=1):
    collisions = []
    check_counter = 0
    i, j = 0, 0
    while i < len(particles1) and j < len(particles2):
        p1 = particles1[i]
        
        # Check particles in particles2 within the x-threshold
        while j < len(particles2) and particles2[j].x < p1.x - threshold:
            j += 1
        
        k = j
        while k < len(particles2) and particles2[k].x <= p1.x + threshold:
            p2 = particles2[k]
            check_counter += 1
            if abs(p1.y - p2.y) <= threshold:
                collisions.append((p1, p2))
            k += 1
        
        i += 1
    
    return collisions, check_counter

# Check for collisions between the particles and each of the adjacent lists
all_collisions = []
total_checks = 0
for other_particles in adjacent:
    collisions, checks = check_collisions(particles, other_particles)
    all_collisions.append(collisions)
    total_checks += checks

# Print all collisions and the total number of checks
for idx, collisions in enumerate(all_collisions):
    print(f"Collisions with particles{idx+2}:")
    for collision in collisions:
        print(collision)

print(f"Total collision checks performed: {total_checks}")
