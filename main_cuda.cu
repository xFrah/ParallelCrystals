#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
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

// CUDA kernel to modify arr2 based on arr1
__global__ void modify_arr2(int *arr1, int *arr2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr2[idx] += arr1[idx];
    }
}

// CUDA kernel for the inner loop operations
__global__ void swap_elements(int *arr1, int *arr2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (arr1[i] == 1) {
            // Skip if arr1[i] == 1
            return;
        }
        for (int j = i; j > 0; j--) {
            if (j == 0 || arr1[j - 1] == 1) {
                if (arr2[j] < arr2[j - 1]) {
                    // Swap elements
                    int temp = arr2[j];
                    arr2[j] = arr2[j - 1];
                    arr2[j - 1] = temp;

                    temp = arr1[j];
                    arr1[j] = arr1[j - 1];
                    arr1[j - 1] = temp;
                } else {
                    break;
                }
            }
        }
    }
}

void random_move(std::vector<int> &arr1, std::vector<int> &arr2) {
    for (size_t i = 0; i < arr2.size(); ++i) {
        arr1[i] = (rand() % 2 == 0) ? -1 : 1;
        arr2[i] += arr1[i];
    }
}

int main() {
    const int n = 10000;
    const int iterations = 1000; // Number of iterations to measure performance
    std::vector<int> arr2(n);
    std::vector<int> arr1(n);

    // Allocate device memory
    int *d_arr1, *d_arr2;
    cudaMalloc(&d_arr1, n * sizeof(int));
    cudaMalloc(&d_arr2, n * sizeof(int));

    // Initialize arr2 with random values and sort it
    for (int i = 0; i < n; ++i) {
        arr2[i] = rand() % 20 + 2;
    }
    std::sort(arr2.begin(), arr2.end());

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        random_move(arr1, arr2);

        // Copy data to device
        cudaMemcpy(d_arr1, arr1.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr2, arr2.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        // Define block and grid sizes
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        // Launch kernel to modify arr2
        modify_arr2<<<gridSize, blockSize>>>(d_arr1, d_arr2, n);
        cudaDeviceSynchronize();

        // Launch kernel to perform the inner loop operations
        swap_elements<<<gridSize, blockSize>>>(d_arr1, d_arr2, n);
        cudaDeviceSynchronize();

        // Copy results back to host
        cudaMemcpy(arr1.data(), d_arr1, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(arr2.data(), d_arr2, n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double iterations_per_second = iterations / elapsed.count();

    // Print results
    for (int i = 0; i < n; ++i) {
        std::cout << arr1[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < n; ++i) {
        std::cout << arr2[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);

    // Print results
    std::cout << "Iterations per second: " << iterations_per_second << std::endl;

    return 0;
}
