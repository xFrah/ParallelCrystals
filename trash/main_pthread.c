#include "cJSON.h"
#include "socket_server.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Helper macro to get JSON values and check their existence
#define GET_JSON_INT_VALUE(JSON_OBJ, NAME) ({                      \
    cJSON *item = cJSON_GetObjectItem(JSON_OBJ, NAME);             \
    if (!item) {                                                   \
        fprintf(stderr, "Missing configuration item: %s\n", NAME); \
        cJSON_Delete(JSON_OBJ);                                    \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
    item->valueint;                                                \
})

int PORT;
int WIDTH;
int HEIGHT;
int NUM_PARTICLES;
int PARTICLE_RADIUS;
int SEED;
int NUM_THREADS;
int SLICE_LENGTH;

struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
};

struct ThreadArgs {
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int thread_id;
    int *thread_counter;
    int start;
    int end;
};

struct Particle *particles; // This should be a pointer
struct Particle **particles_x;
struct Particle **particles_temp_x;

int socket_holder;

void get_configuration() {
    FILE *f = fopen("config.json", "r");
    if (f == NULL) {
        printf("Error configuration opening file\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *string = malloc(fsize + 1);
    fread(string, 1, fsize, f);
    fclose(f);
    string[fsize] = 0;
    cJSON *json = cJSON_Parse(string);
    if (json == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
            cJSON_Delete(json); // Cleanup JSON object
        }
        exit(1);
    }

    PORT = GET_JSON_INT_VALUE(json, "network_port");
    HEIGHT = GET_JSON_INT_VALUE(json, "screen_height");
    WIDTH = GET_JSON_INT_VALUE(json, "screen_width");
    NUM_PARTICLES = GET_JSON_INT_VALUE(json, "num_particles");
    PARTICLE_RADIUS = GET_JSON_INT_VALUE(json, "particle_radius");
    SEED = GET_JSON_INT_VALUE(json, "seed");
    NUM_THREADS = GET_JSON_INT_VALUE(json, "num_threads");
    SLICE_LENGTH = NUM_PARTICLES / NUM_THREADS;

    cJSON_Delete(json);
}

int compare_particles_x(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a; // we can probably optimize this by offsetting the pointer
    struct Particle *p2 = *(struct Particle **)b;
    return p1->new_x - p2->new_x;
}

int compare_particles_y(const void *a, const void *b) {
    struct Particle *p1 = *(struct Particle **)a;
    struct Particle *p2 = *(struct Particle **)b;
    return p1->new_y - p2->new_y;
}

void init_particles() {
    srand(SEED);
    particles = malloc(NUM_PARTICLES * sizeof(struct Particle));     // Allocating space for an array of struct Particle
    particles_x = malloc(NUM_PARTICLES * sizeof(struct Particle *)); // Allocating space for an array of pointers
    particles_temp_x = malloc(NUM_PARTICLES * sizeof(struct Particle *));

    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = rand() % WIDTH;
        particles[i].y = rand() % HEIGHT;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
        particles[i].walker = rand() % 6 ? 1 : 0;
        particles_x[i] = &particles[i];
        particles_temp_x[i] = &particles[i];
    }
    qsort(particles_x, NUM_PARTICLES, sizeof(struct Particle *), compare_particles_x); // first sort
}

void move_particles() {
    for (int i = 0; i < NUM_PARTICLES; i++) { // move particles
        if (particles[i].walker) {
            particles[i].new_x = particles[i].x + 1 + (-2 * (rand() % 2));
            particles[i].new_y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
    }
}

void update_particles() {
    for (int i = 0; i < NUM_PARTICLES; i++) { // move particles
        particles[i].x = particles[i].new_x;
        particles[i].y = particles[i].new_y;
    }
}

void barrier(int *thread_counter, pthread_mutex_t *mutex, pthread_cond_t *cond, int thread_id) {
    pthread_mutex_lock(mutex);
    *thread_counter = *thread_counter + 1;
    if (*thread_counter == NUM_THREADS + 1) { // +1 for main thread
        *thread_counter = 0;
        memcpy(particles_x, particles_temp_x, NUM_PARTICLES * sizeof(struct Particle *)); // sort on temp is finished, copy to main
        // copy particles to old_particles
        update_particles(); // new particle positions are ready, update structs
        // printf("%d> Barrier reached (%d/%d)\n", thread_id, num_threads + 1, num_threads + 1);
        socket_server_send(socket_holder, particles, NUM_PARTICLES * sizeof(struct Particle));
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
    int j;
    struct Particle *p1;
    struct Particle *p2;
    while (1) {
        barrier(args->thread_counter, args->mutex, args->cond, args->thread_id);
        // printf("%d> Started\n", args->thread_id);
        for (int i = args->start; i < args->end; i++) {
            j = i + 1;
            p1 = particles_x[i];
            while (j < NUM_PARTICLES && abs(particles_x[j]->x - particles_x[i]->x) <= PARTICLE_RADIUS) {
                p2 = particles_x[j];
                j++;
                if (p1->walker == p2->walker) {
                    continue;
                }

                if (abs(p2->y - p1->y) <= PARTICLE_RADIUS) {
                    p1->walker = 0;
                    p2->walker = 0;
                }
            }
        }
    }
    return NULL;
}

int main() {
    get_configuration();
    socket_holder = socket_server_start(PORT);
    printf("DEBUG 1\n");
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int thread_counter = 0;
    pthread_t tid[NUM_THREADS];
    struct ThreadArgs args[NUM_THREADS];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    init_particles();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].mutex = &mutex;
        args[i].cond = &cond;
        args[i].thread_counter = &thread_counter;
        args[i].thread_id = i + 1;
        args[i].start = i * SLICE_LENGTH;
        args[i].end = (i + 1) * SLICE_LENGTH;
        pthread_create(&tid[i], NULL, array_slice_thread, &args[i]);
    }
    while (1) {
        move_particles();
        qsort(particles_temp_x, NUM_PARTICLES, sizeof(struct Particle *), compare_particles_x); // TODO assign 2 threads to sort
        barrier(&thread_counter, &mutex, &cond, 0);
    }
    return 0;
}