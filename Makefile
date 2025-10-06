# Makefile for Mandelbrot MPI Project
# Large Scale Modelling - Project 1

# Default values
N ?= 1
CHUNK_SIZE ?= 10
SCHEDULE ?= static
COMMUNICATION ?= blocking

.PHONY: baseline test test-all clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  baseline                    - Generate baseline data arrays"
	@echo "  test                        - Run single test combination"
	@echo "  test-all                    - Run all test combinations"
	@echo "  clean                       - Clean generated files"
	@echo "  help                        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make baseline SIZE=100x100                    # Generate baseline for specific size"
	@echo "  make test SCHEDULE=static COMMUNICATION=blocking N=4 CHUNK_SIZE=20"
	@echo "  make test-all N=2 CHUNK_SIZE=15              # All combinations with 2 processes"

# Generate baseline data arrays
baseline:
	@echo "Generating baseline data arrays..."
	./tests/generate_baseline.sh $(SIZE)

# Single test target with parameters
test:
	@echo "Testing $(SCHEDULE) + $(COMMUNICATION) with $(N) MPI processes (chunk_size=$(CHUNK_SIZE)) against all baselines..."
	./tests/test_runner.sh $(N) $(SCHEDULE) $(COMMUNICATION) $(CHUNK_SIZE)

# Run all test combinations
test-all:
	@echo "Running all test combinations with $(N) MPI processes (chunk_size=$(CHUNK_SIZE))..."
	$(MAKE) test SCHEDULE=static COMMUNICATION=blocking N=$(N) CHUNK_SIZE=$(CHUNK_SIZE)
	$(MAKE) test SCHEDULE=static COMMUNICATION=nonblocking N=$(N) CHUNK_SIZE=$(CHUNK_SIZE)
	$(MAKE) test SCHEDULE=dynamic COMMUNICATION=blocking N=$(N) CHUNK_SIZE=$(CHUNK_SIZE)
	$(MAKE) test SCHEDULE=dynamic COMMUNICATION=nonblocking N=$(N) CHUNK_SIZE=$(CHUNK_SIZE)
	@echo "All tests completed successfully for all sizes!"


# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.png" -delete
	find . -name "*.jpg" -delete
	find . -name "*.jpeg" -delete
	find . -name "*.npy" -delete
	find . -name "*.csv" -delete
	find . -name "*.pdf" -delete
	rm -rf tests/baseline_data/*
	rm -rf experiments/*
	rm -rf Plots/*