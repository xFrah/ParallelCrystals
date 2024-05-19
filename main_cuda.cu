#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <iostream>

using namespace cooperative_groups;

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

Particle *particles;
Particle **particles_x;
Particle **particles_y;
Particle **particles_temp_x;
Particle **particles_temp_y;

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
        particles[i].x = rand() % WIDTH;
        particles[i].y = rand() % HEIGHT;
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


__global__ void cooperativeKernel(Particle *particles, Particle **particles_x, Particle **particles_y, Particle **particles_temp_x, Particle **particles_temp_y, int num_particles, int particle_radius) {
    // Get the cooperative group for the entire grid
    grid_group grid = this_grid();
    
    // Perform some simple computation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d\n", idx);
    int iteration = 0;
    if (idx == 0) {
        curandState state;
        curand_init(123123, 0, 0, &state);
        while (true) {
            iteration++;
            // printf("Iteration %d\n", iteration);
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
            iteration++;
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
                            printf("Collision between %d and %d at iteration %d\n", p1->id, p2->id, iteration);
                        }
                    }
                }
            }
            grid.sync();
            grid.sync();
        }
    }
}

void queryDevices() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }
}


int main() {
    queryDevices();
    allocate_memory();
    init_particles();

    // Define kernel launch parameters
    int numBlocks = 2;
    int threadsPerBlock = 32;

    // Check if the device supports cooperative launch
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch.\n";
        return -1;
    }

    void *kernelArgs[] = { &particles,&particles_x, &particles_y, &particles_temp_x, &particles_temp_y, &NUM_PARTICLES, &PARTICLE_RADIUS };

    // Launch the cooperative kernel
    cudaError_t err = cudaLaunchCooperativeKernel((void*)cooperativeKernel, numBlocks, threadsPerBlock, kernelArgs);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch cooperative kernel: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    std::cout << "Cooperative kernel executed successfully.\n";
    return 0;
}
