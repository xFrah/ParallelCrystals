#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace cg = cooperative_groups;
__device__ curandState state;

#define CHECK_CUDA_ERROR(call) {                                          \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                  << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
        exit(err);                                                        \
    }                                                                     \
}


struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
};

// Unified Memory
__managed__ Particle *particles;
__managed__ Particle **particles_x;
__managed__ Particle **particles_y;
__managed__ Particle **particles_temp_x;
__managed__ Particle **particles_temp_y;

int WIDTH = 1000;
int HEIGHT = 1000;
int NUM_PARTICLES = 1000;
int PARTICLE_RADIUS = 5;
int SEED = 1234;
int NUM_THREADS = 10;
int SLICE_LENGTH = 100;

struct compare_particles_x {
    __host__ __device__
    bool operator()(const Particle *a, const Particle *b) const {
        return a->new_x < b->new_x;
    }
};

struct compare_particles_y {
    __host__ __device__
    bool operator()(const Particle *a, const Particle *b) const {
        return a->new_y < b->new_y;
    }
};

void allocate_memory() {
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles_x, NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles_y, NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles_temp_x, NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&particles_temp_y, NUM_PARTICLES * sizeof(Particle *)));
}


void init_particles() {
    // int blockSize = 256;
    // int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    // init_particles_kernel<<<numBlocks, blockSize>>>(particles, NUM_PARTICLES, WIDTH, HEIGHT);
    // cudaDeviceSynchronize();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = i % WIDTH;
        particles[i].y = i % HEIGHT;
        particles[i].walker = rand() % 2;
        particles[i].y_index = i;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
        particles_x[i] = &particles[i];
        particles_y[i] = &particles[i];
        particles_temp_x[i] = &particles[i];
        particles_temp_y[i] = &particles[i];
    }
    std::cout << "Initialized particles" << std::endl;
}


__global__ void persistent_kernel(Particle **particles_x, Particle **particles_y, Particle **particles_temp_x, Particle **particles_temp_y, int num_particles, int particle_radius) {
    printf("DEBUG 3.5\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cg::grid_group grid = cg::this_grid();
    if (idx == 0) {
        while (true) {
            for (int i = 0; i < num_particles; i++) { // move particles
                if (particles_x[i]->walker) {
                    particles_x[i]->new_x = particles_x[i]->x + 1 + (-2 * (curand(&state) % 2));
                    particles_x[i]->new_y = particles_x[i]->y + 1 + (-2 * (curand(&state) % 2));
                }
            }

            thrust::device_ptr<Particle*> dev_ptr_x(particles_temp_x);
            thrust::sort(dev_ptr_x, dev_ptr_x + num_particles, compare_particles_x());

            thrust::device_ptr<Particle*> dev_ptr_y(particles_temp_y);
            thrust::sort(dev_ptr_y, dev_ptr_y + num_particles, compare_particles_y());
            // Synchronize all threads in the grid
            grid.sync();

            // cudaMemcpy(particles_x, particles_temp_x, num_particles * sizeof(Particle *), cudaMemcpyDeviceToDevice);
            // cudaMemcpy(particles_y, particles_temp_y, num_particles * sizeof(Particle *), cudaMemcpyDeviceToDevice);

            // same but we are in device, so we can't use cudaMemcpy
            for (int i = 0; i < num_particles; i++) {
                particles_x[i] = particles_temp_x[i];
                particles_y[i] = particles_temp_y[i];
            }

            for (int i = 0; i < num_particles; i++) { // update particles
                particles[i].x = particles[i].new_x;
                particles[i].y = particles[i].new_y;
                particles_y[i]->y_index = i;
            }

            // Synchronize again to ensure all threads wait for the operation to complete
            grid.sync();
        }
    } else {
        int num_threads = blockDim.x * gridDim.x;
        // Calculate the portion of the array this thread should process
        int start = (idx * num_particles) / num_threads;
        int end = ((idx + 1) * num_particles) / num_threads;
        char found;
        int j, k, s, i;
        Particle *p1, *p2, *pk;
        while (true) {
            for (i = start; i < end; i++) {
                j = i + 1;
                while (j < num_particles && abs(particles_x[j]->x - particles_x[i]->x) <= particle_radius) {
                    p1 = particles_x[i];
                    p2 = particles_x[j];
                    found = 0;
                    j++;
                    if (p1->walker == p2->walker) {
                        continue;
                    }

                    s = (p1->y_index < p2->y_index) ? 1 : -1;
                    k = p1->y_index + s;

                    while (!found && k < num_particles && abs(particles_y[k]->y - p1->y) <= particle_radius) {
                        pk = particles_y[k];
                        k = k + s;

                        if (p1->y_index == pk->y_index) {
                            for (int i = 0; i < num_particles; i++) {
                                printf("%d ", particles[i].y_index);
                            }
                            printf("\n");
                        }

                        if (p2->id == pk->id) {
                            found = 1;
                            p1->walker = 0;
                            p2->walker = 0;
                        }
                    }
                }
            }
            grid.sync();
            grid.sync();
        }
    }
}

void launch_persistent_kernel() {
    int blockSize = 256; // Number of threads per block
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    std::cout << "DEBUG 3.1" << std::endl;

    void *kernelArgs[] = { &particles_x, &particles_y, &particles_temp_x, &particles_temp_y, &NUM_PARTICLES, &PARTICLE_RADIUS };
    std::cout << "DEBUG 3.2" << std::endl;

    // Check if device supports cooperative groups
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch" << std::endl;
        return;
    } else {
        std::cout << "Device supports cooperative launch" << std::endl;
    }

    // print all args before launching for debug
    // print persistent kernel address
    std::cout << "persistent_kernel address: " << (void*)persistent_kernel << std::endl;
    // print numBlocks
    std::cout << "numBlocks: " << numBlocks << std::endl;
    // print blockSize
    std::cout << "blockSize: " << blockSize << std::endl;
    // print kernelArgs
    for (int i = 0; i < sizeof(kernelArgs) / sizeof(kernelArgs[0]); i++) {
        std::cout << "kernelArgs[" << i << "]: " << kernelArgs[i] << std::endl;
    }
    cudaError_t err = cudaLaunchCooperativeKernel((void*)persistent_kernel, numBlocks, blockSize, kernelArgs);
    if (err != cudaSuccess) {
        std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
    }
    std::cout << "DEBUG 3.3" << std::endl;
    cudaDeviceSynchronize();
    std::cout << "DEBUG 3.4" << std::endl;
}

int main() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << cudaGetErrorString(error);
    } else {
        std::cout << "No CUDA error detected" << std::endl;
    }
    std::cout << "DEBUG 1" << std::endl;
    allocate_memory();
    std::cout << "DEBUG 2" << std::endl;
    init_particles();
    std::cout << "DEBUG 3" << std::endl;

    launch_persistent_kernel();
    std::cout << "DEBUG 4" << std::endl;

    while (1) {
        std::cout << "DEBUG 5" << std::endl;
    }

    return 0;
}
