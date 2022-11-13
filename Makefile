SHELL := /bin/bash -O extglob

CXX := g++
CXXFLAGS := -g -std=c++1z -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS := -fopenmp

VPATH = src:tests/unit

FILES := main.cc gf.hh fmatrix.hh packed_fmatrix.hh

BIN := ff-det ff-det-VPC ff-det-VPC-PAR ff-det-PAR ff-det-test ff-det-test-VPC

all: $(BIN)

ff-det: $(FILES)
	$(CXX) src/main.cc $(CXXFLAGS) -o $@

ff-det-VPC: $(FILES)
	$(CXX) src/main.cc $(CXXFLAGS) -D VPC=1 -o $@

ff-det-VPC-PAR: $(FILES)
	$(CXX) src/main.cc $(CXXFLAGS) -D VPC=1 -D PAR=1 -o $@

ff-det-PAR: $(FILES)
	$(CXX) src/main.cc $(CXXFLAGS) -D PAR=1 -o $@

ff-det-test: gf_test.o fmatrix_test.o tests.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-test-VPC: gf_testVPC.o fmatrix_testVPC.o tests.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ff-det-test-512: gf_test512.o fmatrix_test512.o tests.o
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
	$(CXX) -S $(CXXFLAGS) -fverbose-asm $^

%VPC.s: %.cc
	$(CXX) -S $(CXXFLAGS) -D VPC=1 -fverbose-asm $^ -o $@

%.asm1: %.s
	c++filt < $^ > $@

%.asm2: %16.o
	objdump -d -S $^ > $@


##################
# OBJECT RECIPES #
##################

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^

%VPC.o: %.cc
	$(CXX) $(CXXFLAGS) -D VPC=1 -c -o $@ $^

%512.o: %.cc
	$(CXX) $(CXXFLAGS) -D AVX512=1 -c -o $@ $^
