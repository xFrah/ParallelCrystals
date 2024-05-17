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
    cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));
    cudaMallocManaged(&particles_x, NUM_PARTICLES * sizeof(Particle *));
    cudaMallocManaged(&particles_y, NUM_PARTICLES * sizeof(Particle *));
    cudaMallocManaged(&particles_temp_x, NUM_PARTICLES * sizeof(Particle *));
    cudaMallocManaged(&particles_temp_y, NUM_PARTICLES * sizeof(Particle *));
}

__global__ void init_particles_kernel(Particle *particles, int n, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(3423423, i, 0, &state);
    if (i < n) {
        particles[i].id = i;
        particles[i].x = i % width;
        particles[i].y = i % height;
        // particles[i].walker = rand() % 2;
        particles[i].walker = curand(&state) % 2;
        particles[i].y_index = i;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
    }
}

void init_particles() {
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    init_particles_kernel<<<numBlocks, blockSize>>>(particles, NUM_PARTICLES, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
}


__global__ void persistent_kernel(Particle **particles_x, Particle **particles_y, Particle **particles_temp_x, Particle **particles_temp_y, int num_particles, int particle_radius) {
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

    void *kernelArgs[] = { &particles_x, &particles_y, &particles_temp_x, &particles_temp_y, &NUM_PARTICLES, &PARTICLE_RADIUS };

    cudaLaunchCooperativeKernel((void*)persistent_kernel, numBlocks, blockSize, kernelArgs);
    cudaDeviceSynchronize();
}

int main() {
    std::cout << "DEBUG 1" << std::endl;
    allocate_memory();
    init_particles();

    launch_persistent_kernel();

    while (1) {
    }

    return 0;
}
