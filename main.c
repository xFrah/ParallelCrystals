#include <stdio.h>
#include <stdlib.h>

#define WIDTH 700
#define HEIGHT 600
#define num_particles 100
#define particle_radius 5
#define seed 17

struct Particle {
    int id;
    int x;
    int y;
    char walker;
    int y_index;
};

struct Particle particles[num_particles];
struct Particle *particles_x[num_particles];
struct Particle *particles_y[num_particles];

void init_particles() {
    srand(seed);
    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = rand() % WIDTH;
        particles[i].y = rand() % HEIGHT;
        particles[i].walker = rand() % 2;
        particles[i].y_index = i;
        particles_x[i] = &particles[i];
        particles_y[i] = &particles[i];
    }
}

void move_particles() {
    for (int i = 0; i < num_particles; i++) {
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (rand() % 2));
        } else {
            particles[i].y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
    }
}

void set_particle_y_index() {
    for (int i = 0; i < num_particles; i++) {
        particles_y[i]->y_index = i;
    }
}

int compare_particles_x(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a;
    struct Particle *p2 = *(struct Particle **)b;
    return p1->x - p2->x;
}

int compare_particles_y(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a;
    struct Particle *p2 = *(struct Particle **)b;
    return p1->y - p2->y;
}


int main() {
    init_particles();
    while (1) {
        qsort(particles_x, num_particles, sizeof(struct Particle *), compare_particles_x);
        qsort(particles_y, num_particles, sizeof(struct Particle *), compare_particles_y);
        set_particle_y_index();
        move_particles();
        char found;
        int j;
        int k;
        struct Particle *p1;
        struct Particle *p2;
        struct Particle *pk;
        for (int i = 0; i < num_particles - 1; i++) {
            j = i + 1;
            while (j < num_particles && abs(particles_x[j]->x - particles_x[i]->x) <= particle_radius) {
                p1 = particles_x[i];
                p2 = particles_x[j];
                found = 0;
                j++;
                if ((!p1->walker && !p2->walker) || (p1->walker && p2->walker)) {
                    continue;
                }

                k = p1->y_index + 1;
                while (!found && k < num_particles && abs(particles_y[k]->y - p1->y) <= particle_radius) {
                    pk = particles_y[k];
                    k++;
                    if (p2->id == pk->id && p2->walker != pk->walker) {
                        found = 1;
                        p1->walker = 0;
                        p2->walker = 0;
                    }
                }

                k = p1->y_index - 1;
                while (!found && k >= 0 && abs(particles_y[k]->y - p1->y) <= particle_radius) {
                    pk = particles_y[k];
                    k--;
                    if (p2->id == pk->id && p2->walker != pk->walker) {
                        found = 1;
                        p1->walker = 0;
                        p2->walker = 0;
                    }
                }
            }
        }
    }
    return 0;
}