SHELL := /bin/bash -O extglob

CXX := g++
CXXFLAGS := -g -std=c++1z -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS := -fopenmp

VPATH = src:tests/unit

BIN := ff-det ff-det-VPC ff-det-PAR ff-det-test ff-det-test-VPC

all: $(BIN)

ff-det: main.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-VPC: mainVPC.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-PAR: mainPAR.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-test: gf_test.o fmatrix_test.o tests.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-test-VPC: gf_testVPC.o fmatrix_testVPC.o tests.o
	$(CXX) $^ -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(BIN)
	rm -f *.o *.s *.asm1 *.asm2

test: ff-det-test
	./ff-det-test -gm -d30 -t10000

#############
# ASM STUFF #
#############

%.s: %.cc
	$(CXX) -D GF2_bits=16 -S $(CXXFLAGS) -fverbose-asm $^

%.asm1: %.s
	c++filt < $^ > $@

%.asm2: %16.o
	objdump -d -S $^ > $@


##################
# OBJECT RECIPES #
##################

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^

%PAR.o: %.cc
	$(CXX) $(CXXFLAGS) -D PAR=1 -c -o $@ $^

%VPC.o: %.cc
	$(CXX) $(CXXFLAGS) -D VPC=1 -c -o $@ $^
