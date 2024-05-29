#include <omp.h>
#include <windows.h>

#include <chrono>
#include <iostream>

#include "cJSON.h"
#include "win_socket_server.h"

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

struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
    struct Particle* next_particle;
};

struct Particle_compatibility {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
};

struct ListHead {
    struct Particle* head;
};

struct Configuration {
    int PORT;
    int WIDTH;
    int HEIGHT;
    int NUM_PARTICLES;
    int PARTICLE_RADIUS;
    int SEED;
    int NUM_THREADS;
    int SLICE_LENGTH;
    int CELL_SIZE;
};

struct Configuration config;
int gridHeight;
int gridWidth;
int cellSize;
struct Particle* particles;
struct ListHead*** d_grid;

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


#define CHECK_CUDA_ERROR(call)                                                                                               \
    {                                                                                                                        \
        cudaError_t err = call;                                                                                              \
        if (err != cudaSuccess) {                                                                                            \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                                                                       \
        }                                                                                                                    \
    }

#define CHECK_LAST_ERROR()                                                                                                   \
    {                                                                                                                        \
        cudaError_t err = cudaGetLastError();                                                                                \
        if (err != cudaSuccess) {                                                                                            \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                                                                       \
        }                                                                                                                    \
    }

void reset_linked_lists() {
    // divide grid between the threads
}

void appendNode(struct ListHead* listHead, struct Particle* newHead) {
    // use pragma critical section to append the node
}

void makeLinkedLists(struct ListHead*** grid, struct Particle* particles) {
    // divide particles between the threads
}
void sort_single_cell_insertionSort(struct ListHead*** grid) {
    // divide grid between the threads and perform insertion sort
}

void check_collisions(struct ListHead* cell1, struct ListHead* cell2) {
    // check for collisions between the particles in the two cells
}

void check_for_collisions(struct ListHead*** grid) {
    // divide grid between the threads
}

// debug function to print all the linked lists in all the cells
void print_linked_lists(struct ListHead*** grid) {
    int counter = 0;

    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            struct ListHead* cell = grid[i][j];
            if (cell->head == NULL) {
                // printf("Cell (%d, %d) is empty\n", i, j);
                continue;
            }
            struct Particle* p = cell->head;
            printf("[%d] Cell (%d, %d): ", counter, i, j);
            while (p != NULL) {
                printf("%d (%d, %d) ", p->id, p->x, p->y);
                counter++;
                p = p->next_particle;
            }
            printf("\n");
        }
    }
}

void move_particles_kernel(struct Particle* particles, struct curandState* states) {
    // divide particles between the threads
}

void init_particles_kernel(struct Particle* particles, struct curandState* states) {
    // divide particles between the threads
}

void initializeGrid(struct ListHead*** grid) {

}

allocate_memory() {
    // Allocate memory for the grid on the host
    struct ListHead*** h_grid = (struct ListHead***)malloc(gridHeight * sizeof(struct ListHead**));
    for (int i = 0; i < gridHeight; ++i) {
        h_grid[i] = (struct ListHead**)malloc(gridWidth * sizeof(struct ListHead*));
    }

    // Allocate memory for the grid on the device
    struct ListHead*** d_grid;
    cudaMalloc(&d_grid, gridHeight_host * sizeof(ListHead**));
    for (int i = 0; i < gridHeight_host; ++i) {
        ListHead** d_row;
        cudaMalloc(&d_row, gridWidth_host * sizeof(ListHead*));
        cudaMemcpy(&d_grid[i], &d_row, sizeof(ListHead*), cudaMemcpyHostToDevice);
    }
}

int main() {
    config = get_configuration();
    cellSize_host = config.CELL_SIZE;
    gridHeight_host = config.HEIGHT / cellSize_host;
    gridWidth_host = config.WIDTH / cellSize_host;

    ListHead*** d_grid = allocate_memory();

    int socket_holder = socket_server_start(config.PORT);
    if (socket_holder < 0) {
        std::cerr << "Failed to start socket server\n";
        return -1;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((gridWidth_host + threadsPerBlock.x - 1) / threadsPerBlock.x, (gridHeight_host + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initializeGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
    init_particles_kernel<<<1, 1>>>(particles, states);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    Particle* local_particles = (Particle*)malloc(config.NUM_PARTICLES * sizeof(Particle));
    Particle_compatibility* local_particles_compatibility = (Particle_compatibility*)malloc(config.NUM_PARTICLES * sizeof(Particle_compatibility));

    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;

    while (1) {
        makeLinkedLists<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles);
        cudaDeviceSynchronize();

        sort_single_cell_insertionSort<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        check_for_collisions<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        move_particles_kernel<<<numBlocks_, threadsPerBlock_>>>(particles, states);
        reset_linked_lists<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        if (iteration % 1000 == 0) {
            cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
            for (int i = 0; i < config.NUM_PARTICLES; i++) {
                local_particles_compatibility[i].id = local_particles[i].id;
                local_particles_compatibility[i].x = local_particles[i].x;
                local_particles_compatibility[i].y = local_particles[i].y;
                local_particles_compatibility[i].walker = local_particles[i].walker;
                local_particles_compatibility[i].y_index = local_particles[i].y_index;
                local_particles_compatibility[i].new_x = local_particles[i].x;
                local_particles_compatibility[i].new_y = local_particles[i].y;
            }
            socket_server_send(socket_holder, local_particles_compatibility, config.NUM_PARTICLES * sizeof(struct Particle_compatibility));
        }
        cudaDeviceSynchronize();

        // printf("Iteration %d\n", iteration++);
        if (iteration++ % 1000 == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << 1000.0 / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
        }
    }

    return 0;
}
