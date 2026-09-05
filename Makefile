CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
# Statically link the C++ runtime. Without this a MinGW binary can pick up a
# different libstdc++ from PATH at run time (Git for Windows ships one) and
# crash. Static linking also makes the executable self-contained.
LDFLAGS  ?= -static-libstdc++ -static-libgcc
BIN      := bin

COMMON := src/matrix.cpp src/network.cpp src/data.cpp

.PHONY: all train gradcheck run check clean

all: $(BIN)/train $(BIN)/gradcheck

$(BIN):
	mkdir -p $(BIN)

$(BIN)/train: $(COMMON) src/main.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/gradcheck: $(COMMON) src/gradcheck.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: $(BIN)/train
	./$(BIN)/train

check: $(BIN)/gradcheck
	./$(BIN)/gradcheck

clean:
	rm -rf $(BIN)
