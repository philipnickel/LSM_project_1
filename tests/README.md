# Tests

## Test Suite: 90 lines, 65 tests, ~19 seconds

### Files

```
tests/
├── test_configs.yaml (12 lines)      # 64 test configurations
├── run_mpi_and_save.py (27 lines)    # MPI helper with error handling  
├── test_correctness.py (34 lines)    # 64 numerical correctness tests
└── test_e2e.py (17 lines)            # 1 end-to-end test
```

### Run All Tests

```bash
pytest tests/
```

**That's it!** Single command runs everything.

### What's Tested

**Numerical Correctness (64 tests)**
- 32 configs with n_ranks=1
- 32 configs with n_ranks=4
- All via subprocess with `mpirun`
- Compares MPI result vs baseline pixel-by-pixel

**Test Coverage Matrix:**
- ✓ 2 chunk sizes (3, 7)
- ✓ 2 image sizes (15x15, 20x20)
- ✓ 2 schedules (static, dynamic)
- ✓ 2 communication modes (blocking, nonblocking)
- ✓ 2 Mandelbrot regions (standard + zoomed)
- ✓ 2 rank counts (1, 4)

**End-to-End (1 test)**
- Runs full TESTS suite via main.py
- Verifies complete pipeline works without errors

### Error Handling

All scripts return proper exit codes:
- `0` = success
- `1` = failure (with stderr message)

Tests check return codes and show full output on failure.

### Design Philosophy

**Minimal, practical, comprehensive**
- No pytest-mpi complexity
- No hardcoded values
- No redundant tests
- Spawns MPI via subprocess (works everywhere)
- Tests what actually matters in production


