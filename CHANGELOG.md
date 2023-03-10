# Change Log for PyTorch

## PyTorch for ROCm5.5 (upstream commit: 36ba2ce)

### Added
- hipSolver integration
- Sync updates from hipify-torch
- MIOpen header files to installation package for ROCm
- Only install miopen-hip-gfx packages for ROCm5.5 onwards
- PyTorch wheels now support using MIOpen kdb files
- Conda packages for ROCm and PyTorch are available on repo.radeon.com.

### Fixed
- Revert to using default conda environment for pytorch build and test
