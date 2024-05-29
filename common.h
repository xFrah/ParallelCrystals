#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include "libs/cJSON.h"  // Make sure to include the cJSON library header if it's used

// Struct definitions
typedef struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
    struct Particle* next_particle;  // Use 'struct' for self-referential structures
} Particle;

typedef struct Particle_compatibility {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
} Particle_compatibility;

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

// Function declarations
Configuration get_configuration();

// Inline function to get JSON values and check their existence
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
