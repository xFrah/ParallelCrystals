## For compiling: 
```bash
gcc main.c socket_server.c cJSON.c -o a.out -Wall -lpthread -g
```

## For running:
First change the configuration in the `config.json` file.

For running the "backend":
```bash
./a.out
```

For running the "frontend":
```bash
python3 displayer.py
```


## Lil diagram

![diagram](diagram.png)