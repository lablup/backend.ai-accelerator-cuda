import subprocess
import logging
from pprint import pformat

from ai.backend.agent.accelerator import accelerator_types
from .gpu import CUDAAccelerator

log = logging.getLogger('ai.backend.accelerator.cuda')


async def init(etcd):
    try:
        ret = subprocess.run(['nvidia-docker', 'version'],
                             stdout=subprocess.PIPE)
    except FileNotFoundError:
        log.info('nvidia-docker is not installed.')
        return 0
    rx = CUDAAccelerator.rx_nvdocker_version
    for line in ret.stdout.decode().strip().splitlines():
        m = rx.search(line)
        if m is not None:
            CUDAAccelerator.nvdocker_version = tuple(map(int, m.group(1).split('.')))
    accelerator_types['cuda'] = CUDAAccelerator
    detected_devices = CUDAAccelerator.list_devices()
    log.info('detected devices:\n' +
             pformat(detected_devices))
