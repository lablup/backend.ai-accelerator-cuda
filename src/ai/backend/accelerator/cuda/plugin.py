import subprocess
import logging
from pprint import pformat
import re

from ai.backend.agent.accelerator import accelerator_types
from ai.backend.common.logging import BraceStyleAdapter
from .gpu import CUDAAccelerator

log = BraceStyleAdapter(logging.getLogger('ai.backend.accelerator.cuda'))


async def init(etcd):
    try:
        ret = subprocess.run(['nvidia-docker', 'version'],
                             stdout=subprocess.PIPE)
    except FileNotFoundError:
        log.warning('nvidia-docker is not installed.')
        log.info('CUDA acceleration is disabled.')
        return 0
    rx = re.compile(r'^NVIDIA Docker: (\d+\.\d+\.\d+)')
    for line in ret.stdout.decode().strip().splitlines():
        m = rx.search(line)
        if m is not None:
            CUDAAccelerator.nvdocker_version = tuple(map(int, m.group(1).split('.')))
            break
    else:
        log.error('could not detect nvidia-docker version!')
        log.info('CUDA acceleration is disabled.')
        return
    accelerator_types['cuda'] = CUDAAccelerator
    detected_devices = CUDAAccelerator.list_devices()
    log.info('detected devices:\n' + pformat(detected_devices))
    log.info('nvidia-docker version: {}', CUDAAccelerator.nvdocker_version)
    log.info('CUDA acceleration is enabled.')
