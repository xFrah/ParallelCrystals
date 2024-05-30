#include "common.h"

int get_json_int_value(cJSON* json_obj, const char* name) {
    cJSON* item = cJSON_GetObjectItem(json_obj, name);
    if (!item) {
        fprintf(stderr, "Missing configuration item: %s\n", name);
        cJSON_Delete(json_obj);
        exit(EXIT_FAILURE);
    }
    return item->valueint;
}

Configuration get_configuration() {
    FILE* f = fopen("config.json", "r");
    if (f == NULL) {
        printf("Error opening configuration file\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* string = (char*)malloc(fsize + 1);
    fread(string, 1, fsize, f);
    fclose(f);
    string[fsize] = 0;
    cJSON* json = cJSON_Parse(string);
    free(string);
    if (json == NULL) {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        exit(1);
    }

    Configuration config;
    config.PORT = get_json_int_value(json, "network_port");
    config.HEIGHT = get_json_int_value(json, "screen_height");
    config.WIDTH = get_json_int_value(json, "screen_width");
    config.NUM_PARTICLES = get_json_int_value(json, "num_particles");
    config.PARTICLE_RADIUS = get_json_int_value(json, "particle_radius");
    config.SEED = get_json_int_value(json, "seed");
    config.NUM_THREADS = get_json_int_value(json, "num_threads");
    config.CELL_SIZE = get_json_int_value(json, "cell_size");
    config.TARGET_DISPLAY_FPS = get_json_int_value(json, "target_display_fps");
    config.SHOW_VISUALLY = get_json_int_value(json, "show_visually");
    config.USE_MAX_HARDWARE_THREADS = get_json_int_value(json, "use_max_hardware_threads");
    config.SLICE_LENGTH = config.NUM_PARTICLES / config.NUM_THREADS;

    cJSON_Delete(json);

    return config;
}
