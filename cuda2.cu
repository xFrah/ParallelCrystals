#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <windows.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>

// Define kernel launch parameters
#define numBlocks_ 68
#define threadsPerBlock_ 32

extern "C" {
    #include "cJSON.h"
    #include "win_socket_server.h"
}

struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
    Particle *next_particle;
};

struct ListHead {
    Particle* head;
    Particle* tail;
    int mutex;
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
};

curandState *states;
__device__ Configuration d_config;
struct Configuration config;
Particle *particles;

extern "C" struct Configuration get_configuration();

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

struct Configuration get_configuration() {
    FILE *f = fopen("config.json", "r");
    if (f == NULL) {
        printf("Error configuration opening file\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *string = (char*)malloc(fsize + 1); // Explicit cast to char*
    fread(string, 1, fsize, f);
    fclose(f);
    string[fsize] = 0;
    cJSON *json = cJSON_Parse(string);
    free(string); // Free the allocated memory
    if (json == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        exit(1);
    }

    cJSON_Delete(json);

    struct Configuration config;
    config.PORT = get_json_int_value(json, "network_port");
    config.HEIGHT = get_json_int_value(json, "screen_height");
    config.WIDTH = get_json_int_value(json, "screen_width");
    config.NUM_PARTICLES = get_json_int_value(json, "num_particles");
    config.PARTICLE_RADIUS = get_json_int_value(json, "particle_radius");
    config.SEED = get_json_int_value(json, "seed");
    config.NUM_THREADS = get_json_int_value(json, "num_threads");
    config.SLICE_LENGTH = config.NUM_PARTICLES / config.NUM_THREADS;
    return config;
}

#define CHECK_CUDA_ERROR(call) {                                          \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                  << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
        exit(err);                                                        \
    }                                                                     \
}

#define CHECK_LAST_ERROR() {                                              \
    cudaError_t err = cudaGetLastError();                                 \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                  << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
        exit(err);                                                        \
    }                                                                     \
}

__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}

__device__ void appendNode(ListHead* listHead, Particle* particle) {
    lock(&(listHead->mutex));

    if (listHead->head == NULL) {
        listHead->head = particle;
        printf("Particle %d is the first in the list\n", particle->id);
    }

    listHead->tail = particle;

    unlock(&(listHead->mutex));
}

__global__ void traverseAndPrintParticles(ListHead*** grid, Particle* particles, int numParticles, int width, int height, int cellSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    int gridX, gridY;
    Particle *p;

    for (int i = start; i < end; i++) {
        p = &particles[i];
        gridX = p->x / cellSize;
        gridY = p->y / cellSize;

        if (gridX < width / cellSize && gridY < height / cellSize) {
            appendNode(grid[gridY][gridX], p);
        } else {
            printf("Particle %d is out of bounds\n", p->id);
        }
    }
}

__global__ void init_particles_kernel(Particle *particles, curandState *states) {
    for (int i = 0; i < numBlocks_ * threadsPerBlock_; i++) {
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    for (int i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
    }
}

__global__ void initializeGrid(ListHead*** grid, int height, int width, int cellSize) {
    int gridHeight = height / cellSize;
    int gridWidth = width / cellSize;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        grid[row][col] = (ListHead*)malloc(sizeof(ListHead));
        grid[row][col]->head = NULL;
        grid[row][col]->tail = NULL;
        grid[row][col]->mutex = 0;
    }
}


int main() {
    config = get_configuration();
    // Copy configuration to device
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks_ * threadsPerBlock_ * sizeof(curandState)));

    int cellSize = config.PARTICLE_RADIUS * 2;

    int gridHeight = config.HEIGHT / cellSize;
    int gridWidth = config.WIDTH / cellSize;

    std::cout << "Grid size: " << gridHeight << "x" << gridWidth << std::endl;

    // Allocate memory for the grid on the host
    ListHead*** h_grid = (ListHead***)malloc(gridHeight * sizeof(ListHead**));
    for (int i = 0; i < gridHeight; ++i) {
        h_grid[i] = (ListHead**)malloc(gridWidth * sizeof(ListHead*));
    }

    std::cout << "Allocated memory for the grid on the host" << std::endl;

    // Allocate memory for the grid on the device
    ListHead*** d_grid;
    cudaMalloc(&d_grid, gridHeight * sizeof(ListHead**));
    for (int i = 0; i < gridHeight; ++i) {
        ListHead** d_row;
        cudaMalloc(&d_row, gridWidth * sizeof(ListHead*));
        cudaMemcpy(&d_grid[i], &d_row, sizeof(ListHead*), cudaMemcpyHostToDevice);
    }

    std::cout << "Allocated memory for the grid on the device" << std::endl;

    // Initialize the grid with linked list heads
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((gridWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (gridHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initializeGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid, config.HEIGHT, config.WIDTH, cellSize);
    cudaDeviceSynchronize();

    std::cout << "Initialized the grid with linked list heads" << std::endl;

    // Initialize particles
    init_particles_kernel<<<1, 1>>>(particles, states);
    cudaDeviceSynchronize();

    std::cout << "Initialized particles" << std::endl;

    // Traverse particles and append them to the linked list in the corresponding cell
    traverseAndPrintParticles<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles, config.NUM_PARTICLES, config.WIDTH, config.HEIGHT, cellSize);
    cudaDeviceSynchronize();

    std::cout << "Traversed particles and appended them to the linked list in the corresponding cell" << std::endl;

    // Free device memory
    for (int i = 0; i < gridHeight; ++i) {
        ListHead** d_row;
        cudaMemcpy(&d_row, &d_grid[i], sizeof(ListHead*), cudaMemcpyDeviceToHost);
        cudaFree(d_row);
    }
    cudaFree(d_grid);

    std::cout << "Freed device memory" << std::endl;

    // Free host memory
    for (int i = 0; i < gridHeight; ++i) {
        free(h_grid[i]);
    }
    free(h_grid);

    std::cout << "Freed host memory" << std::endl;

    return 0;
}
