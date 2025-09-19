.PHONY: all clean run

# Makefile for MLP training with LibTorch
CXX = nvcc
CXXFLAGS = -std=c++17 -O3

ARCH_GEN = -gencode arch=compute_90,code=sm_90

# TORCH_DIR ?= ../libtorch
# TORCH_DIR = /users/lsx/.local/lib/python3.10/site-packages/torch
# TORCH_DIR = /users/lsx/pytorch/torch

JSON_INCLUDE = /usr/include/nlohmann
# CUDA_HOME ?= /usr/local/cuda-12.6


INCLUDES = -I$(JSON_INCLUDE)


# rpaths so the executable can find shared libs without LD_LIBRARY_PATH
# LDFLAGS   = -Wl,-rpath,$(TORCH_DIR)/lib \
#             -Wl,-rpath,$(CUDA_HOME)/lib64 \
# 			-Wl,--no-as-needed


# Libraries to link
LIBS      = -lcudart -lnvToolsExt -lpthread -ldl \
			-lcublas -lcublasLt



# Output directory
BIN_DIR = ./bin

# Default target
all: $(BIN_DIR) native

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile the MLP training program
native: native.cu | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $< $(LIBS) $(LDFLAGS) $(ARCH_GEN)

native_pipe: native_pipe.cu | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $< $(LIBS) $(LDFLAGS) $(ARCH_GEN)

# Compile the MLP training program
gh: gh.cu | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $< $(LIBS) $(LDFLAGS) $(ARCH_GEN)

test: test.cu | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BIN_DIR)/$@ $< $(LIBS) $(LDFLAGS) $(ARCH_GEN)


# Clean build artifacts
clean:
	rm -f $(BIN_DIR)/native

# Run the program'
ARGS ?= 4096 4096 1.0 5
run_native: $(BIN_DIR) | $(BIN_DIR)/native 
	./$(BIN_DIR)/native $(ARGS)

run_native_pipe: $(BIN_DIR) | $(BIN_DIR)/native_pipe
	./$(BIN_DIR)/native_pipe $(ARGS)

run_gh: $(BIN_DIR) | $(BIN_DIR)/gh
	./$(BIN_DIR)/gh $(ARGS)

run_test: $(BIN_DIR) | $(BIN_DIR)/test
	./$(BIN_DIR)/gh

# Run the program
nsys_native: $(BIN_DIR) | $(BIN_DIR)/native 
	nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true -f true --output ./profile/native ./bin/native $(ARGS)

nsys_native_pipe: $(BIN_DIR) | $(BIN_DIR)/native_pipe
	nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true -f true --output ./profile/native_pipe ./bin/native_pipe $(ARGS)


nsys_gh: $(BIN_DIR) | $(BIN_DIR)/gh 
	nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true -f true --output ./profile/gh ./bin/gh $(ARGS)

nsys_test: $(BIN_DIR) | $(BIN_DIR)/test
	nsys profile --trace=cuda,nvlink --show-output=true --cuda-memory-usage=true --sample=cpu --output ./profile/test ./bin/test

ncu_native: $(BIN_DIR) | $(BIN_DIR)/native
	ncu -f -o ./profile/native --section MemoryWorkloadAnalysis_Chart ./bin/native $(ARGS)

ncu_gh: $(BIN_DIR) | $(BIN_DIR)/gh 
# 	OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ncu -f -o ./profile/gh --section MemoryWorkloadAnalysis_Chart ./bin/gh
# 	OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ncu --section C2CLink --replay-mode range --nvtx --nvtx-include "Lora/" -f -o ./profile/gh --section C2CLink ./bin/gh
# 	ncu --section C2CLink --replay-mode kernel --nvtx --nvtx-include "Lora/" -f -o ./profile/gh --section C2CLink ./bin/gh $(ARGS)
	ncu --section C2CLink --replay-mode kernel -f -o ./profile/gh --section C2CLink ./bin/gh $(ARGS)

ncu_test: $(BIN_DIR) | $(BIN_DIR)/test 
# 	ncu --section MemoryWorkloadAnalysis_Chart --section C2CLink -f -o ./profile/test ./bin/test
	ncu --section C2CLink -f -o ./profile/test ./bin/test
# 	ncu -f -o ./profile/test --set full ./bin/test


