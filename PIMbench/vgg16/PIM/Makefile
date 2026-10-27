# Makefile: C++ version of vgg16 and test_conv
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

PROJ_ROOT = ../../..
include ${PROJ_ROOT}/Makefile.common

CXXFLAGS += -O3 -march=native

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
    CXXFLAGS += -fopenmp
endif

COMPILE_WITH_JPEG ?= 0
ifeq ($(COMPILE_WITH_JPEG), 1)
    CXXFLAGS += -DCOMPILE_WITH_JPEG
    # Add linker flags for libjpeg
    LDFLAGS += -ljpeg
endif

# Define both executables
EXECS := vgg16.out test_conv.out

debug perf dramsim3_integ: $(EXECS)

# Rule for vgg16
vgg16.out: vgg16.cpp $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

# Rule for test_conv
test_conv.out: test_conv.cpp $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -rf $(EXECS) *.dSYM