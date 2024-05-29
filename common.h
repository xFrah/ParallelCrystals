#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include "libs/cJSON.h"

// Struct definitions
typedef struct Particle {
    int id;
    int x;
    int y;
    int walker;
    struct Particle* next_particle;
} Particle;

typedef struct ListHead {
    Particle* head;
} ListHead;

typedef struct Configuration {
    int PORT;
    int WIDTH;
    int HEIGHT;
    int NUM_PARTICLES;
    int PARTICLE_RADIUS;
    int SEED;
    int NUM_THREADS;
    int SLICE_LENGTH;
    int CELL_SIZE;
    int TARGET_DISPLAY_FPS;
} Configuration;

Configuration get_configuration();

inline int get_json_int_value(cJSON* json_obj, const char* name) {
    cJSON* item = cJSON_GetObjectItem(json_obj, name);
    if (!item) {
        fprintf(stderr, "Missing configuration item: %s\n", name);
        cJSON_Delete(json_obj);
        exit(EXIT_FAILURE);
    }
    return item->valueint;
}

#endif // COMMON_H
