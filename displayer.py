import time
import pygame
import sys
import socket
import json
import struct
import threading


def load_config():
    with open("config.json", "r") as file:
        config = json.load(file)

    width = config["screen_width"]
    height = config["screen_height"]
    particle_radius = config["particle_radius"]
    port = config["network_port"]
    num_particles = config["num_particles"]
    return width, height, particle_radius, port, num_particles


# Constants
WIDTH, HEIGHT, PARTICLE_RADIUS, PORT, NUM_PARTICLES = load_config()
gray = (200, 200, 200)
black = (0, 0, 0)
dark_red = (128, 0, 0)
max_length = (NUM_PARTICLES * 64) + (NUM_PARTICLES * 0.2)
packet_counter = 0
start_counter = time.time()
current_packet = {}

# Define the packet and particle structure formats
header_format = "4s"
length_format = "i"
footer_format = "4s"
particle_format = "iiiiq"

header_size = struct.calcsize(header_format)
length_size = struct.calcsize(length_format)
footer_size = struct.calcsize(footer_format)
particle_size = struct.calcsize(particle_format)


class Particle:
    def __init__(self, iid, x, y, walker, _):
        self.id = iid
        self.x = x
        self.y = y
        self.walker = walker

def connect_to_server(address):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(address)
            print("Connected to server successfully.")
            global WIDTH, HEIGHT, PARTICLE_RADIUS, PORT, NUM_PARTICLES
            WIDTH, HEIGHT, PARTICLE_RADIUS, PORT, NUM_PARTICLES = load_config()
            return sock
        except ConnectionRefusedError:
            print("Connection failed. Trying again in 5 seconds...")
            time.sleep(5)


def receive_and_parse_packets(sock):
    global current_packet, packet_counter, start_counter
    buffer = bytearray()
    while True:
        try:
            data = sock.recv(50000)
            if not data:
                print("Connection closed by the server.")
                break

            buffer += data

            while True:
                if b"PART" not in buffer:
                    break
                start_index = buffer.find(b"PART")
                del buffer[:start_index]

                if len(buffer) < header_size + length_size:
                    break

                length_offset = header_size
                (length,) = struct.unpack(length_format, buffer[length_offset : length_offset + length_size])
                if length > max_length:
                    buffer = bytearray()
                    break

                packet_end = length_offset + length_size + length + footer_size
                if len(buffer) < packet_end:
                    break

                footer = buffer[packet_end - footer_size : packet_end]
                if footer != b"ENDP":
                    break

                particles_data = buffer[length_offset + length_size : packet_end - footer_size]
                current_packet = {
                    struct.unpack(particle_format, particles_data[i : i + particle_size])[0]: Particle(
                        *struct.unpack(particle_format, particles_data[i : i + particle_size])
                    )
                    for i in range(0, length, particle_size)
                }
                packet_counter += 1
                if packet_counter % 1000 == 0:
                    print(f"Packet rate: {1000 / (time.time() - start_counter):.2f} packets per second, {len(current_packet)} particles per packet.")
                    start_counter = time.time()
                buffer = buffer[packet_end:]

        except Exception as e:
            print("Error occurred:", e)
            break


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random Walk")

    clock = pygame.time.Clock()
    server_address = ("localhost", PORT)

    while True:
        sock = connect_to_server(server_address)
        thread = threading.Thread(target=receive_and_parse_packets, args=(sock,), daemon=True)
        thread.start()

        while thread.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill(black)

            for _, particle in current_packet.items():
                pygame.draw.circle(
                    screen,
                    gray if particle.walker else dark_red,
                    (particle.x, particle.y),
                    PARTICLE_RADIUS,
                    PARTICLE_RADIUS,
                )

            pygame.display.update()
            clock.tick(60)

        print("Attempting to reconnect...")
        time.sleep(5)  # Wait a bit before reconnecting


if __name__ == "__main__":
    main()
