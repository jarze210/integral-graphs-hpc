CXX = g++
NVCC = nvcc
EIGEN_INC = -I/usr/include/eigen3
CXXFLAGS = -O3 $(EIGEN_INC)
OMPFLAGS = -fopenmp
AVXFLAGS = -mavx2 -mfma
NVCCFLAGS = -O3

BIN_DIR = bin

all: seq omp avx cuda_classic cuda_cyclic

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

seq: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) src/sequential.cpp -o $(BIN_DIR)/sequential

omp: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) src/openmp.cpp -o $(BIN_DIR)/openmp

avx: $(BIN_DIR)
	$(CXX) -O3 $(OMPFLAGS) $(AVXFLAGS) src/openmp_avx2.cpp -o $(BIN_DIR)/openmp_avx2

cuda_classic: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/cuda_jacobi_classic.cu -o $(BIN_DIR)/cuda_jacobi_classic

cuda_cyclic: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/cuda_jacobi_cyclic.cu -o $(BIN_DIR)/cuda_jacobi_cyclic

clean:
	rm -rf $(BIN_DIR)
