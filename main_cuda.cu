#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <windows.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>

extern "C" {
    #include "cJSON.h"
    #include "win_socket_server.h"
}


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

// Define kernel launch parameters
#define numBlocks 68
#define threadsPerBlock 32

struct Particle {
    int id;
    int x;
    int y;
    int walker;
    int y_index;
    int new_x;
    int new_y;
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
Particle **particles_x;
Particle **particles_temp_x;

extern "C" struct Configuration get_configuration();

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
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_x, config.NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_temp_x, config.NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks * threadsPerBlock * sizeof(curandState)));
}

__global__ void init_particles_kernel(Particle *particles, Particle **particles_x, Particle **particles_temp_x, curandState *states) {
    for (int i = 0; i < numBlocks * threadsPerBlock; i++) {
        printf("Initializing state %d\n", i);
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    for (int i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
        particles_x[i] = &particles[i];
        particles_temp_x[i] = &particles[i];
    }
}

__global__ void move_particles_kernel(Particle **particles_x, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (int i = start; i < end; i++) { // move particles
        if (particles_x[i]->walker) {
            particles_x[i]->new_x = particles_x[i]->x + 1 + (-2 * (curand(&states[idx]) % 2));
            particles_x[i]->new_y = particles_x[i]->y + 1 + (-2 * (curand(&states[idx]) % 2));
        }
    }
}

void init_particles() {
    init_particles_kernel<<<1, 1>>>(particles, particles_x, particles_temp_x, states);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
}

__global__ void collision_check_kernel(Particle *particles, Particle **particles_x, Particle **particles_temp_x) {
    int num_particles = d_config.NUM_PARTICLES;
    int particle_radius = d_config.PARTICLE_RADIUS;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    // Calculate the portion of the array this thread should process
    int start = (idx * num_particles) / num_threads;
    int end = ((idx + 1) * num_particles) / num_threads;
    int j, i;
    Particle *p1, *p2;

    for (i = start; i < end; i++) {
        j = i + 1;
        while (j < num_particles && abs(particles_x[j]->x - particles_x[i]->x) <= particle_radius) {
            p1 = particles_x[i];
            p2 = particles_x[j];
            j++;
            if (p1->walker == p2->walker) {
                continue;
            }

            if (abs(p2->y - p1->y) <= particle_radius) {
                p1->walker = 0;
                p2->walker = 0;
            }
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


__global__ void update_particles_kernel(Particle *particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (int i = start; i < end; i++) { // update particles
        particles[i].x = particles[i].new_x;
        particles[i].y = particles[i].new_y;
    }
}


int main() {
    config = get_configuration();
    // Copy configuration to device
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));

    // Start the socket server
    int socket_holder = socket_server_start(config.PORT);
    if (socket_holder < 0) {
        std::cerr << "Failed to start socket server\n";
        return -1;
    }

    queryDevices();
    allocate_memory();
    init_particles();

    int iteration = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // Local particles array to send to the client
    Particle * local_particles = (Particle *)malloc(config.NUM_PARTICLES * sizeof(Particle));

    while (true) {
        collision_check_kernel<<<numBlocks, threadsPerBlock>>>(particles, particles_x, particles_temp_x);
        move_particles_kernel<<<numBlocks, threadsPerBlock>>>(particles_x, states);
        cudaDeviceSynchronize();

        thrust::sort(thrust::device, particles_temp_x, particles_temp_x + config.NUM_PARTICLES, compare_particles_x());
        cudaDeviceSynchronize();

        cudaMemcpy(particles_x, particles_temp_x, config.NUM_PARTICLES * sizeof(Particle *), cudaMemcpyDeviceToDevice);

        update_particles_kernel<<<numBlocks, threadsPerBlock>>>(particles);
        cudaDeviceSynchronize();

        if (iteration++ % 200 == 0) {
            cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
            socket_server_send(socket_holder, local_particles, config.NUM_PARTICLES * sizeof(struct Particle));
        }
        if (iteration % 1000 == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << 1000.0 / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
        }
    }

    std::cout << "Cooperative kernel executed successfully.\n";
    return 0;
}
