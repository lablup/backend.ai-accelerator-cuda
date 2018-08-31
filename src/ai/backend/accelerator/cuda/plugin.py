from ai.backend.agent.accelerator import accelerator_types
from .gpu import CUDAAccelerator


async def init():
    accelerator_types['cuda'] = CUDAAccelerator
