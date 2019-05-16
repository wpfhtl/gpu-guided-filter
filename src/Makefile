CC=g++
NVCC=nvcc
CUDAFLAGS= -arch=sm_61 -cudart=shared -rdc=true -G -g
LDLIBS=`pkg-config opencv --cflags --libs`
BIN=guided_filter
SRC=guided_filter.cu box_filter.cu
all: $(BIN)

$(BIN):
	$(NVCC) $(SRC) $(CUDAFLAGS) $(LDLIBS) -Iincludes -o $(BIN)

clean:
	$(RM) $(BIN)
	$(RM) *.o
