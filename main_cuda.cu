#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <windows.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <chrono>
#include <iostream>

// Define kernel launch parameters
#define numBlocks_ 68
#define threadsPerBlock_ 32

extern "C" {
#include "common.h"
#include "libs/cJSON.h"
#include "libs/win_socket_server.h"
}

curandState *states;
__device__ Configuration d_config;
__device__ uint64_t gridHeight;
__device__ uint64_t gridWidth;
__device__ uint64_t cellSize;
struct Configuration config;
uint64_t gridHeight_host;
uint64_t gridWidth_host;
uint64_t cellSize_host;
Particle *particles;
__device__ uint64_t *cellStartIndices;

#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

#define CHECK_LAST_ERROR()                                                    \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

__device__ void normalizeGridPosition(int *gridX, int *gridY) {
    if (*gridX >= gridWidth) *gridX = gridWidth - 1;
    if (*gridY >= gridHeight) *gridY = gridHeight - 1;
    if (*gridX < 0) *gridX = 0;
    if (*gridY < 0) *gridY = 0;
}

__device__ int fromCelltoFlattenedIndex(int x, int y) {
    return gridWidth * y + x;
}

__device__ int cellFromCoords(int x, int y, int *gridX, int *gridY) {
    *gridX = x / cellSize;
    *gridY = y / cellSize;
    normalizeGridPosition(gridX, gridY);
    return fromCelltoFlattenedIndex(*gridX, *gridY);
}

__device__ int getCellLength(int index) {
    if (index == gridHeight * gridWidth - 1) {
        return d_config.NUM_PARTICLES - cellStartIndices[index];
    }
    return cellStartIndices[index + 1] - cellStartIndices[index];
}


__device__ void check_collisions(Particle* particles, int gridX1, int gridY1, int gridX2, int gridY2) {
    int threshold = d_config.PARTICLE_RADIUS;

    int cell1 = fromCelltoFlattenedIndex(gridX1, gridY1);
    int cell2 = fromCelltoFlattenedIndex(gridX2, gridY2);
    int length1 = getCellLength(cell1);
    int length2 = getCellLength(cell2);
    int start1 = cellStartIndices[cell1];
    int start2 = cellStartIndices[cell2];
    Particle* particles1 = &particles[start1];
    Particle* particles2 = &particles[start2];

    for (int i = 0; i < length1; ++i) {
        Particle* p1 = &particles1[i];
        for (int j = 0; j < length2; ++j) {
            Particle* pj = &particles2[j];
            if (pj->x < p1->x - threshold) {
                continue;
            }
            if (pj->x > p1->x + threshold) {
                break;
            }
            if (p1->walker != pj->walker && abs(p1->y - pj->y) <= threshold) {
                // Handle collision
                // printf("Collision between particles %d and %d\n", p1->id, pj->id);
                p1->walker = 0;
                pj->walker = 0;
            }
        }
    }
}

struct ParticleComparator {
    __device__ bool operator()(const Particle &a, const Particle &b) const {
        // Compare by grid index, then by x in case of parity
        if (a.index == b.index) {
            return a.x < b.x;
        }
        return a.index < b.index;
    }
};

__global__ void checkSortedAndInitializeCellStartIndices(Particle *d_array) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;

    for (uint64_t i = start; i < end; i++) {
        int gridX, gridY;
        int cellIndex = cellFromCoords(d_array[i].x, d_array[i].y, &gridX, &gridY);
        int prevGridX = -1;
        int prevGridY = -1;
        if (i > 0) {
            int prevCellIndex = cellFromCoords(d_array[i - 1].x, d_array[i - 1].y, &prevGridX, &prevGridY);
        }
        if (i == 0 || (gridX != prevGridX || gridY != prevGridY)) {
            // printf("Thread %llu: Cell index: %d, (%d, %d), Start: %llu\n", idx, cellIndex, gridX, gridY, i);
            cellStartIndices[cellIndex] = i;
        }
        d_array[i].index = cellIndex;
    }
}

// kernel to print particle array for debugging
__global__ void print_particles(Particle *particles) {
    for (int i = 0; i < 200; i++) {
        int gridX, gridY;
        int cellIndex = cellFromCoords(particles[i].x, particles[i].y, &gridX, &gridY);
        // print cell, x, y and id
        printf("[%d] Cell: %d, x: %d, y: %d, id: %d (%d, %d)\n", i, cellIndex, particles[i].x, particles[i].y, particles[i].id, gridX, gridY);
    }
}

__global__ void check_for_collisions(Particle *particles) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        // check what kind of cell this is
        if (row == 0 && col == 0) {
            // top left corner: compare with cell at the right, cell below, cell at bottom right
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col + 1, row + 1);
            check_collisions(particles, col, row, col, row);
        } else if (row == 0 && col == gridWidth - 1) {
            // top right corner: compare with cell below and cell at bottom left
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col - 1, row + 1);
            check_collisions(particles, col, row, col, row);
        } else if (row == gridHeight - 1 && col == 0) {
            // bottom left corner: compare with cell at the right
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col, row);
        } else if (row == gridHeight - 1 && col == gridWidth - 1) {
            // bottom right corner: compare with no one
            check_collisions(particles, col, row, col, row);
        } else if (row == 0) {
            // top edge: compare with cell at the right, cell at bottom left, cell below and cell at bottom right
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col - 1, row + 1);
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col + 1, row + 1);
            check_collisions(particles, col, row, col, row);
        } else if (row == gridHeight - 1) {
            // bottom edge: compare with cell at the right
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col, row);
        } else if (col == 0) {
            // left edge: compare with cell at the right, cell below, cell at bottom right
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col + 1, row + 1);
            check_collisions(particles, col, row, col, row);
        } else if (col == gridWidth - 1) {
            // right edge: compare with cell below and cell at bottom left
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col - 1, row + 1);
            check_collisions(particles, col, row, col, row);
        } else {
            // middle: compare with cell at the right, cell below, cell at bottom right, cell at bottom left
            check_collisions(particles, col, row, col + 1, row);
            check_collisions(particles, col, row, col, row + 1);
            check_collisions(particles, col, row, col + 1, row + 1);
            check_collisions(particles, col, row, col - 1, row + 1);
            check_collisions(particles, col, row, col, row);
        }
    }
}

__global__ void move_particles_kernel(Particle *particles, curandState *states) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (uint64_t i = start; i < end; i++) { // move particles
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (curand(&states[idx]) % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (curand(&states[idx]) % 2));
        }
        particles[i].next_particle = NULL;
    }
}

__global__ void init_particles_kernel(Particle *particles, curandState *states) {
    for (uint64_t i = 0; i < numBlocks_ * threadsPerBlock_; i++) {
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    // allocate space for cell start indices
    cellStartIndices = (uint64_t *)malloc(gridHeight * gridWidth * sizeof(uint64_t));
    for (int i = 0; i < gridHeight * gridWidth; i++) {
        cellStartIndices[i] = 0;
    }
    for (uint64_t i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2;
        particles[i].next_particle = NULL;
    }
}

void allocate_memory() {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridHeight, &gridHeight_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridWidth, &gridWidth_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cellSize, &cellSize_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks_ * threadsPerBlock_ * sizeof(curandState)));
}

int main() {
    config = get_configuration();
    cellSize_host = config.CELL_SIZE;
    gridHeight_host = config.HEIGHT / cellSize_host;
    gridWidth_host = config.WIDTH / cellSize_host;

    allocate_memory();

    int socket_holder;
    if (config.SHOW_VISUALLY) {
        std::cout << "Starting socket server on port " << config.PORT << "\n";
        socket_holder = socket_server_start(config.PORT);
        if (socket_holder < 0) {
            std::cerr << "Failed to start socket server\n";
            return -1;
        }
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((gridWidth_host + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (gridHeight_host + threadsPerBlock.y - 1) / threadsPerBlock.y);
    init_particles_kernel<<<1, 1>>>(particles, states);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    std::cout << "Initialization done\n";

    Particle *local_particles = (Particle *)malloc(config.NUM_PARTICLES * sizeof(Particle));

    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;

    thrust::device_ptr<Particle> dev_ptr(particles);

    while (1) {
        thrust::sort(dev_ptr, dev_ptr + config.NUM_PARTICLES, ParticleComparator());
        cudaDeviceSynchronize();
        // CHECK_LAST_ERROR();

        // print_particles<<<1, 1>>>(particles);

        // std::cout << "Sorting done\n";

        // Check sorted array and initialize cell start indices
        checkSortedAndInitializeCellStartIndices<<<numBlocks_, threadsPerBlock_>>>(particles);
        cudaDeviceSynchronize();
        // CHECK_LAST_ERROR();

        // check_for_collisions<<<blocksPerGrid, threadsPerBlock>>>(particles);
        // cudaDeviceSynchronize();
        // CHECK_LAST_ERROR();
        
        // std::cout << "Collision check done\n";

        move_particles_kernel<<<numBlocks_, threadsPerBlock_>>>(particles, states);
        cudaDeviceSynchronize();
        // CHECK_LAST_ERROR();

        // std::cout << "Movement done\n";

        auto end = std::chrono::high_resolution_clock::now();
        iteration++;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
            if (config.SHOW_VISUALLY) {
                cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
                socket_server_send(socket_holder, local_particles, config.NUM_PARTICLES * sizeof(struct Particle));
            }
            iteration = 0;
        }
    }

    return 0;
}
