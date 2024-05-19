#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <windows.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <iostream>

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

    // PORT = get_json_int_value(json, "network_port");
    // HEIGHT = get_json_int_value(json, "screen_height");
    // WIDTH = get_json_int_value(json, "screen_width");
    // NUM_PARTICLES = get_json_int_value(json, "num_particles");
    // PARTICLE_RADIUS = get_json_int_value(json, "particle_radius");
    // SEED = get_json_int_value(json, "seed");
    // NUM_THREADS = get_json_int_value(json, "num_threads");
    // SLICE_LENGTH = NUM_PARTICLES / NUM_THREADS;

    cJSON_Delete(json);

    // same but with struct
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

void allocate_memory(struct Configuration config) {
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_x, config.NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_y, config.NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_temp_x, config.NUM_PARTICLES * sizeof(Particle *)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles_temp_y, config.NUM_PARTICLES * sizeof(Particle *)));
}

// init particle kernel
__global__ void init_particles_kernel(Configuration config, Particle *particles, Particle **particles_x, Particle **particles_y, Particle **particles_temp_x, Particle **particles_temp_y) {
    curandState state;
    curand_init(123123, 0, 0, &state);
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&state) % config.WIDTH;
        particles[i].y = curand(&state) % config.HEIGHT;
        particles[i].walker = curand(&state) % 2;
        particles[i].y_index = i;
        particles[i].new_x = particles[i].x;
        particles[i].new_y = particles[i].y;
        particles_x[i] = &particles[i];
        particles_y[i] = &particles[i];
        particles_temp_x[i] = &particles[i];
        particles_temp_y[i] = &particles[i];
    }
}

void init_particles(Configuration config) {
    init_particles_kernel<<<1, 1>>>(config, particles, particles_x, particles_y, particles_temp_x, particles_temp_y);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
}

__global__ void cooperativeKernel(Particle *particles, Particle **particles_x, Particle **particles_y, Particle **particles_temp_x, Particle **particles_temp_y, Configuration *config) {
    int num_particles = config->NUM_PARTICLES;
    int particle_radius = config->PARTICLE_RADIUS;

    // start time
    clock_t start = clock();
    
    // Perform some simple computation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d\n", idx);
    if (idx == 0) {
        curandState state;
        curand_init(123123, 0, 0, &state);

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

    } else {
        int num_threads = blockDim.x * gridDim.x;
        // Calculate the portion of the array this thread should process
        int start = (idx * num_particles) / num_threads;
        int end = ((idx + 1) * num_particles) / num_threads;
        // printf("Thread %d: %d - %d\n", idx, start, end);
        char found;
        int j, k, s, i;
        Particle *p1, *p2, *pk;

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

                    if (p2->id == pk->id) {
                        found = 1;
                        p1->walker = 0;
                        p2->walker = 0;
                        printf("Collision between %d and %d\n", p1->id, p2->id);
                    }
                }
            }
        }
    }
    // printf("Time of %d: %f\n", idx, (double)(clock() - start) / CLOCKS_PER_SEC);
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


// update_particles_kernel
__global__ void update_particles_kernel(Configuration *config, Particle *particles, Particle **particles_y) {
    for (int i = 0; i < config->NUM_PARTICLES; i++) { // update particles
        particles[i].x = particles[i].new_x;
        particles[i].y = particles[i].new_y;
        particles_y[i]->y_index = i;
    }
}


int main() {
    struct Configuration config = get_configuration();

    // // Start the socket server
    int socket_holder = socket_server_start(config.PORT);
    if (socket_holder < 0) {
        std::cerr << "Failed to start socket server\n";
        return -1;
    }
    // copy the configuration to the device
    Configuration *d_config;
    CHECK_CUDA_ERROR(cudaMalloc(&d_config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_config, &config, sizeof(Configuration), cudaMemcpyHostToDevice));
    queryDevices();
    allocate_memory(config);
    init_particles(config);

    // socket_server_send(socket_holder, particles, config.NUM_PARTICLES * sizeof(struct Particle));

    std::cout << "Configuration loaded:\n";
    std::cout << "PORT: " << config.PORT << "\n";
    std::cout << "HEIGHT: " << config.HEIGHT << "\n";
    std::cout << "WIDTH: " << config.WIDTH << "\n";
    std::cout << "NUM_PARTICLES: " << config.NUM_PARTICLES << "\n";
    std::cout << "PARTICLE_RADIUS: " << config.PARTICLE_RADIUS << "\n";
    std::cout << "SEED: " << config.SEED << "\n";
    std::cout << "NUM_THREADS: " << config.NUM_THREADS << "\n";
    std::cout << "SLICE_LENGTH: " << config.SLICE_LENGTH << "\n";

    // Define kernel launch parameters
    int numBlocks = 48;
    int threadsPerBlock = 32;

    // Check if the device supports cooperative launch
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Device does not support cooperative launch.\n";
        return -1;
    } else {
        std::cout << "Device supports cooperative launch.\n";
    }

    int iteration = 0;

    while (true) {
        cooperativeKernel<<<numBlocks, threadsPerBlock>>>(particles, particles_x, particles_y, particles_temp_x, particles_temp_y, d_config);
        cudaDeviceSynchronize();
        CHECK_LAST_ERROR();

        cudaMemcpy(particles_x, particles_temp_x, config.NUM_PARTICLES * sizeof(Particle *), cudaMemcpyDeviceToDevice);
        cudaMemcpy(particles_y, particles_temp_y, config.NUM_PARTICLES * sizeof(Particle *), cudaMemcpyDeviceToDevice);

        update_particles_kernel<<<1, 1>>>(d_config, particles, particles_y);
        cudaDeviceSynchronize();
        CHECK_LAST_ERROR();
        // socket_server_send(socket_holder, particles, config.NUM_PARTICLES * sizeof(struct Particle));
        iteration++;
        if (iteration % 100 == 0) {
            std::cout << "Iteration: " << iteration << "\n";
        }
    }

    std::cout << "Cooperative kernel executed successfully.\n";
    return 0;
}
