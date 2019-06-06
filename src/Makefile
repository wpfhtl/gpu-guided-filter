CC=g++
NVCC=nvcc
CUDAFLAGS= -arch=sm_61 -cudart=shared -rdc=true -G -O3
LDLIBS=`pkg-config opencv --cflags --libs`
BIN=guided_filter
SRC=guided_filter.cu box_filter.cu math_kernels.cu
#SRC=guided_filter_simple.cu box_filter_simple.cu math_kernels.cu
SRC2=bilateral/bilateral.cu math_kernels.cu

all: $(BIN) bilat

$(BIN):
	$(NVCC) $(SRC) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN)

bilat:
	$(NVCC) $(SRC2) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o bilat

clean:
	$(RM) $(BIN) bilat
	$(RM) *.o out.png
