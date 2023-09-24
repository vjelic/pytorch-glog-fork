# Change Log for PyTorch

## PyTorch for ROCm6.0 
- Created from upstream commit: https://github.com/pytorch/pytorch/commit/39ff80125f5c6d240a4b59010253cb4adab5090f
- Creation date - Sep 08 2023
- Torch Version - 2.2
- [TODO] IFU?

### Fixed
- 

## PyTorch for ROCm5.7 (upstream commit: 08c4a44)

### Fixed
- [SWDEV-396381] Fixed FSDP UTs by limiting to 8 GPUs
- Fixed Circular issue in hipify using current_state and iterative DFS.
- Added hipblaslt support. Requires setting the environment variable USE_HIPBLASLT=1 and must be on a supported architecture, for example gfx90a.

