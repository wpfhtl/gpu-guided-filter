CC=g++
NVCC=nvcc
CUDAFLAGS= -arch=sm_61 -cudart=shared -rdc=true -G -O3
LDLIBS=`pkg-config opencv --cflags --libs`
BIN=../build/guided_filter
BIN1=../build/guided_filter_simple
BIN11=../build/guided_filter_grey
BIN12=../build/guided_filter_grey_simple
SRC=guided_filter.cu box_filter.cu math_kernels.cu
SRC1=guided_filter_simple.cu box_filter_simple.cu math_kernels.cu
SRC2=bilateral/bilateral.cu math_kernels.cu

all: $(BIN) $(BIN1) $(BIN11) $(BIN12) bilat

$(BIN):
	$(NVCC) $(SRC) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN)

$(BIN1):
	$(NVCC) $(SRC1) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN1)

$(BIN11):
	$(NVCC) $(SRC1) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN11)

$(BIN12):
	$(NVCC) $(SRC1) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN12)

bilat:
	$(NVCC) $(SRC2) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o bilat

clean:
	$(RM) $(BIN) $(BIN1) $(BIN11) $(BIN12) bilat
	$(RM) *.o out.png
