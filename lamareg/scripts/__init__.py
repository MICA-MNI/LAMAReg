"""LaMAR scripts package."""

# Import main functions to make them available at package level
from lamareg.scripts.lamar import lamareg
from lamareg.scripts import synthseg, coregister, apply_warp

__all__ = [
    'lamareg',
    'synthseg', 
    'coregister',
    'apply_warp',
]