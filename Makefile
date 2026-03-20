#
# Simulation of rainwater flooding
# 
#
# Parallel computing (Degree in Computer Engineering)
# 2024/2025
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp
MPICC=mpicc

# Directories
SRC_DIR=src

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Targets to build
OBJS=flood_seq flood_omp flood_mpi

# Rules. By default show help
help:
	@echo
	@echo "Simulation of rainwater flooding"
	@echo
	@echo "make flood_seq   Build only the sequential version"
	@echo "make flood_omp   Build only the OpenMP version"
	@echo "make flood_mpi   Build only the MPI version"
	@echo
	@echo "make all         Build all versions (Sequential, OpenMP, MPI)"
	@echo "make debug       Build sequential version with debug output"
	@echo "make animation   Build sequential version prepared for the Python animation"
	@echo "make clean       Remove executable files"
	@echo

all: $(OBJS)

flood_seq: $(SRC_DIR)/flood.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

flood_omp: $(SRC_DIR)/flood_omp.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

flood_mpi: $(SRC_DIR)/flood_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	$(MAKE) FLAGS="$(FLAGS) -DDEBUG -g" flood_seq

# Compile to generate animation
animation:
	$(MAKE) FLAGS="$(FLAGS) -DDEBUG -DANIMATION -g" flood_seq