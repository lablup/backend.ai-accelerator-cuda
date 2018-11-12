import subprocess
import logging
from pprint import pformat

from ai.backend.agent.accelerator import accelerator_types
# from .gpu import CUDAAccelerator
from .tpu import TPUAccelerator

log = logging.getLogger('ai.backend.accelerator.tpu')


async def init(etcd):
    try:
        ret = subprocess.run(['ctpu', 'version'], stdout=subprocess.PIPE)
    except FileNotFoundError:
        log.info('ctpu is not installed.')
        return 0
    rx = TPUAccelerator.rx_ctpu_version
    for line in ret.stdout.decode().strip().splitlines():
        m = rx.search(line)
        if m is not None:
            TPUAccelerator.ctpu_version = tuple(map(int, m.group(1).split('.')))
    accelerator_types['tpu'] = TPUAccelerator
    detected_devices = TPUAccelerator.list_devices()
    log.info('detected devices:\n' + pformat(detected_devices))
