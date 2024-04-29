#include "socket_server.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 800
#define HEIGHT 600
#define num_particles 50
#define particle_radius 5
#define seed 17
#define num_threads 10
#define slice (num_particles / num_threads);

struct Particle {
    int id;
    int x;
    int y;
    int old_x;
    int old_y;
    int walker;
    int y_index;
    int old_y_index;
};

struct ThreadArgs {
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int thread_id;
    int *thread_counter;
    int start;
    int end;
};

struct Particle particles[num_particles];
struct Particle *particles_x[num_particles];
struct Particle *particles_y[num_particles];
struct Particle *particles_temp_x[num_particles];
struct Particle *particles_temp_y[num_particles];

int socket_holder;

int compare_particles_x(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a; // we can probably optimize this by offsetting the pointer
    struct Particle *p2 = *(struct Particle **)b;
    return p1->x - p2->x;
}

int compare_particles_y(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a;
    struct Particle *p2 = *(struct Particle **)b;
    return p1->y - p2->y;
}

void init_particles() {
    srand(seed);
    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = rand() % WIDTH;
        particles[i].y = rand() % HEIGHT;
        particles[i].old_x = particles[i].x;
        particles[i].old_y = particles[i].y;
        particles[i].walker = rand() % 6 ? 1 : 0;
        particles_x[i] = &particles[i];
        particles_y[i] = &particles[i];
        particles_temp_x[i] = &particles[i];
        particles_temp_y[i] = &particles[i];
    }
    qsort(particles_x, num_particles, sizeof(struct Particle *), compare_particles_x); // first sort
    qsort(particles_y, num_particles, sizeof(struct Particle *), compare_particles_y);
    for (int i = 0; i < num_particles; i++) { // first y_index update
        particles_y[i]->y_index = i;
        particles_y[i]->old_y_index = i;
    }
}

void update_particles() {
    for (int i = 0; i < num_particles; i++) { // move particles
        particles[i].old_x = particles[i].x;
        particles[i].old_y = particles[i].y;
        particles[i].old_y_index = particles[i].y_index;
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (rand() % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
        particles_y[i]->y_index = i;
    }
}

void barrier(int *thread_counter, pthread_mutex_t *mutex, pthread_cond_t *cond, int thread_id) {
    pthread_mutex_lock(mutex);
    *thread_counter = *thread_counter + 1;
    if (*thread_counter == num_threads + 1) { // +1 for main thread
        *thread_counter = 0;
        memcpy(particles_x, particles_temp_x, sizeof(particles_x)); // sort on temp is finished, copy to main
        memcpy(particles_y, particles_temp_y, sizeof(particles_y));
        // copy particles to old_particles
        update_particles(); // new particle positions are ready, update structs
        // print every particle old_y_index, in the same line
        // for (int i = 0; i < num_particles; i++) {
        //     printf("%d ", particles[i].old_y_index);
        // }
        // printf("\n");
        // printf("%d> Barrier reached (%d/%d)\n", thread_id, num_threads + 1, num_threads + 1);
        socket_server_send(socket_holder, particles, num_particles * sizeof(struct Particle));
        pthread_cond_broadcast(cond);
    } else {
        // printf("%d> Waiting at barrier (%d/%d)\n", thread_id, *thread_counter, num_threads + 1);
        while (pthread_cond_wait(cond, mutex) != 0)
            ;
    }
    pthread_mutex_unlock(mutex);
}

void *array_slice_thread(void *vargp) {
    struct ThreadArgs *args = (struct ThreadArgs *)vargp;
    char found;
    int j;
    int k;
    int s;
    struct Particle *p1;
    struct Particle *p2;
    struct Particle *pk;
    while (1) {
        barrier(args->thread_counter, args->mutex, args->cond, args->thread_id);
        // printf("%d> Started\n", args->thread_id);
        for (int i = args->start; i < args->end; i++) {
            j = i + 1;
            while (j < num_particles && abs(particles_x[j]->old_x - particles_x[i]->old_x) <= particle_radius) {
                p1 = particles_x[i];
                p2 = particles_x[j];
                found = 0;
                j++;
                if (p1->walker == p2->walker) {
                    continue;
                }

                if (p1->old_y_index < p2->old_y_index) {
                    s = 1;
                } else {
                    s = -1;
                }

                k = p1->old_y_index + s;
                // print k
                printf("%d> old_y_index = %d, k = %d, s = %d\n", args->thread_id, p1->old_y_index, k, s);
                while (!found && k < num_particles && abs(particles_y[k]->old_y - p1->old_y) <= particle_radius) {
                    pk = particles_y[k];
                    k = k + s;
                    // if old_y_index of p1 and pk is the same
                    if (p1->old_y_index == pk->old_y_index) {
                        for (int i = 0; i < num_particles; i++) {
                            printf("%d ", particles[i].old_y_index);
                        }
                        printf("\n");
                    }
                    printf("%d p1 = (%d, %d, id=%d, yi=%d), p2 = (%d, %d, id=%d, yi=%d), pk = (%d, %d, id=%d, yi=%d)\n", s, p1->old_x, p1->old_y, p1->id, p1->old_y_index, p2->old_x, p2->old_y, p2->id, p2->old_y_index, pk->old_x, pk->old_y, pk->id, pk->old_y_index);
                    if (p2->id == pk->id) {
                        found = 1;
                        printf("%d> Found collision between %d and %d\n", args->thread_id, p1->id, p2->id);
                        p1->walker = 0;
                        p2->walker = 0;
                    }
                }
            }
        }
    }
    return NULL;
}

int main() {
    socket_holder = socket_server_start();
    printf("DEBUG 1\n");
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int thread_counter = 0;
    pthread_t tid[num_threads];
    struct ThreadArgs args[num_threads];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    init_particles();
    for (int i = 0; i < num_threads; i++) {
        args[i].mutex = &mutex;
        args[i].cond = &cond;
        args[i].thread_counter = &thread_counter;
        args[i].thread_id = i + 1;
        args[i].start = i * slice;
        args[i].end = (i + 1) * slice;
        pthread_create(&tid[i], NULL, array_slice_thread, &args[i]);
    }
    while (1) {
        qsort(particles_temp_x, num_particles, sizeof(struct Particle *), compare_particles_x); // TODO assign 2 threads to sort
        // printf("MAIN> Sorted x\n");
        qsort(particles_temp_y, num_particles, sizeof(struct Particle *), compare_particles_y);
        // printf("MAIN> Sorted y\n");
        barrier(&thread_counter, &mutex, &cond, 0);
    }
    return 0;
}