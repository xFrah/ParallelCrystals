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
__device__ int top_left_corner = 0;
__device__ int top_right_corner = 0;
__device__ int bottom_left_corner = 0;
__device__ int bottom_right_corner = 0;
__device__ int left_edge = 0;
__device__ int right_edge = 0;
__device__ int top_edge = 0;
__device__ int bottom_edge = 0;
__device__ int middle = 0;


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
    }

    listHead->tail = particle;

    unlock(&(listHead->mutex));
}

__global__ void makeLinkedLists(ListHead*** grid, Particle* particles, int numParticles, int width, int height, int cellSize) {
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

__global__ void sort_single_cell_insertionSort(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int cellSize = d_config.PARTICLE_RADIUS * 2;
    int gridHeight = d_config.HEIGHT / cellSize;
    int gridWidth = d_config.WIDTH / cellSize;

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        Particle* head = cell->head;

        if (head == NULL) {
            // printf("Cell (%d, %d) is empty\n", row, col);
            return;  // If the cell is empty, there's nothing to sort.
        }

        // Insertion sort on the linked list by x-coordinate
        Particle* sorted = NULL;
        Particle* current = head;
        while (current != NULL) {
            Particle* next = current->next_particle;
            if (sorted == NULL || current->x < sorted->x) {
                current->next_particle = sorted;
                sorted = current;
            } else {
                Particle* search = sorted;
                while (search->next_particle != NULL && search->next_particle->x < current->x) {
                    search = search->next_particle;
                }
                current->next_particle = search->next_particle;
                search->next_particle = current;
            }
            current = next;
        }

        // Update the cell's head to the new sorted list head
        cell->head = sorted;

        // Update the tail to the last particle in the sorted list
        Particle* tail = sorted;
        while (tail->next_particle != NULL) {
            tail = tail->next_particle;
        }
        cell->tail = tail;
    }
}

// def check_collisions(particles1, particles2, threshold=1):
// collisions = []
// check_counter = 0
// i, j = 0, 0
// while i < len(particles1) and j < len(particles2):
//     p1 = particles1[i]
        
//     # Check particles in particles2 within the x-threshold
//     while j < len(particles2) and particles2[j].x < p1.x - threshold:
//         j += 1
        
//     k = j
//     while k < len(particles2) and particles2[k].x <= p1.x + threshold:
//         p2 = particles2[k]
//         check_counter += 1
//         if abs(p1.y - p2.y) <= threshold:
//             collisions.append((p1, p2))
//         k += 1
        
//     i += 1
    
// return collisions, check_counter
__device__ void check_collisions(ListHead *cell1, ListHead *cell2) {
    Particle *p1 = cell1->head;
    Particle *pj = cell2->head;
    int threshold = d_config.PARTICLE_RADIUS;

    while (p1 != NULL && pj != NULL) {
        while (pj != NULL && pj->x < p1->x - threshold) {
            pj = pj->next_particle;
        }

        Particle *pk = pj;
        while (pk != NULL && pk->x <= p1->x + threshold) {
            if (abs(p1->y - pk->y) <= threshold) {
                // Collision detected
            }
            pk = pk->next_particle;
        }
        p1 = p1->next_particle;
    }
}

__global__ void check_for_collisions(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int cellSize = d_config.PARTICLE_RADIUS * 2;
    int gridHeight = d_config.HEIGHT / cellSize;
    int gridWidth = d_config.WIDTH / cellSize;

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        // check what kind of cell this is, top left corner, top right corner, bottom left corner, bottom right corner, left edge, right edge, top edge, bottom edge, or middle
        if (row == 0 && col == 0) {
            // top left corner
            atomicAdd(&top_left_corner, 1);
            // compare with cell at the right, cell below, cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
        } else if (row == 0 && col == gridWidth - 1) {
            // top right corner
            atomicAdd(&top_right_corner, 1);
            // compare with cell below and cell at bottom left
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col - 1]);
        } else if (row == gridHeight - 1 && col == 0) {
            // bottom left corner
           atomicAdd(&bottom_left_corner, 1);
           // compare with cell at the right
            check_collisions(cell, grid[row][col + 1]);
        } else if (row == gridHeight - 1 && col == gridWidth - 1) {
            // bottom right corner
            atomicAdd(&bottom_right_corner, 1);
            // compare with no one
        } else if (row == 0) {
            // top edge
            atomicAdd(&top_edge, 1);
            // compare with cell at the right, cell at bottom left, cell below and cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col - 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
        } else if (row == gridHeight - 1) {
            // bottom edge
            atomicAdd(&bottom_edge, 1);
            // compare with cell at the right
            check_collisions(cell, grid[row][col + 1]);
        } else if (col == 0) {
            // left edge
            atomicAdd(&left_edge, 1);
            // compare with cell at the right, cell below, cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
        } else if (col == gridWidth - 1) {
            // right edge
            atomicAdd(&right_edge, 1);
            // compare with cell below and cell at bottom left
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col - 1]);
        } else {
            // middle
            atomicAdd(&middle, 1);
            // compare with cell at the right, cell below, cell at bottom right, cell at bottom left
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
            check_collisions(cell, grid[row + 1][col - 1]);
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
    makeLinkedLists<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles, config.NUM_PARTICLES, config.WIDTH, config.HEIGHT, cellSize);
    cudaDeviceSynchronize();

    std::cout << "Traversed particles and appended them to the linked list in the corresponding cell" << std::endl;

    // Sort the particles in each cell using insertion sort
    dim3 sortThreadsPerBlock(16, 16);
    dim3 sortBlocksPerGrid((gridWidth + sortThreadsPerBlock.x - 1) / sortThreadsPerBlock.x,
                           (gridHeight + sortThreadsPerBlock.y - 1) / sortThreadsPerBlock.y);
    sort_single_cell_insertionSort<<<sortBlocksPerGrid, sortThreadsPerBlock>>>(d_grid);
    cudaDeviceSynchronize();

    std::cout << "Sorted the particles in each cell using insertion sort" << std::endl;

    // Check for collisions
    check_for_collisions<<<sortBlocksPerGrid, sortThreadsPerBlock>>>(d_grid);
    cudaDeviceSynchronize();

    // copy counters from device to host
    int top_left_corner_host, top_right_corner_host, bottom_left_corner_host, bottom_right_corner_host, left_edge_host, right_edge_host, top_edge_host, bottom_edge_host, middle_host;
    cudaMemcpyFromSymbol(&top_left_corner_host, top_left_corner, sizeof(int));
    cudaMemcpyFromSymbol(&top_right_corner_host, top_right_corner, sizeof(int));
    cudaMemcpyFromSymbol(&bottom_left_corner_host, bottom_left_corner, sizeof(int));
    cudaMemcpyFromSymbol(&bottom_right_corner_host, bottom_right_corner, sizeof(int));
    cudaMemcpyFromSymbol(&left_edge_host, left_edge, sizeof(int));
    cudaMemcpyFromSymbol(&right_edge_host, right_edge, sizeof(int));
    cudaMemcpyFromSymbol(&top_edge_host, top_edge, sizeof(int));
    cudaMemcpyFromSymbol(&bottom_edge_host, bottom_edge, sizeof(int));
    cudaMemcpyFromSymbol(&middle_host, middle, sizeof(int));

    std::cout << "top left corner: " << top_left_corner_host << std::endl;
    std::cout << "top right corner: " << top_right_corner_host << std::endl;
    std::cout << "bottom left corner: " << bottom_left_corner_host << std::endl;
    std::cout << "bottom right corner: " << bottom_right_corner_host << std::endl;
    std::cout << "left edge: " << left_edge_host << std::endl;
    std::cout << "right edge: " << right_edge_host << std::endl;
    std::cout << "top edge: " << top_edge_host << std::endl;
    std::cout << "bottom edge: " << bottom_edge_host << std::endl;
    std::cout << "middle: " << middle_host << std::endl;

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
