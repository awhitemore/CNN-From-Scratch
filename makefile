CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3

SRC = main.cpp tensor.cpp conv_layer.cpp softmax.cpp dense.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = cnn

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^   

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(EXEC)

# Optional: Download Fashion-MNIST data
data:
	@echo "Downloading Fashion-MNIST data..."
	@curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
	@curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz  
	@curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
	@curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
	@gunzip *.gz
	@echo "Data downloaded and extracted!"

.PHONY: all clean data