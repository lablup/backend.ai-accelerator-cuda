from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Collection

import attr
import requests

from ai.backend.agent.accelerator import (
    AbstractAccelerator, AbstractAcceleratorInfo,
)
# from .nvidia import libcudart

log = logging.getLogger('ai.backend.accelerator.tpu')


@attr.s(auto_attribs=True)
class TPUAcceleratorInfo(AbstractAcceleratorInfo):

    # TODO: make this configurable
    unit_memory = (2 * (2 ** 30))  # 1 unit = 2 GiB
    unit_proc = 8                  # 1 unit = 8 SMPs

    def max_share(self) -> Decimal:
        mem_shares = self.memory_size / self.unit_memory
        proc_shares = self.processing_units / self.unit_proc
        common_shares = min(mem_shares, proc_shares)
        quantum = Decimal('.01')
        return Decimal(common_shares).quantize(quantum, ROUND_DOWN)

    def share_to_spec(self, share: Decimal) -> (int, int):
        # TODO: consider the memory margin for heap size?
        return (
            int(self.unit_memory * share),
            int(self.unit_proc * share),
        )

    def spec_to_share(self, requested_memory: int,
                      requested_proc_units: int) -> Decimal:
        mem_share = requested_memory / self.unit_memory
        proc_share = requested_proc_units / self.unit_proc
        required_share = max(mem_share, proc_share)
        quantum = Decimal('.01')
        return Decimal(required_share).quantize(quantum, ROUND_UP)


class TPUAccelerator(AbstractAccelerator):

    slot_key = 'tpu'  # TODO: generalize

    ctpu_version = (0, 0)
    rx_ctpu_version = re.compile(r'^ctpu version: (\d+\.\d+)')

    num_devices = 0

    @classmethod
    def list_devices(cls) -> Collection[TPUAcceleratorInfo]:
        ret = subprocess.run(['ctpu', 'ls', '-no-header', '|', 'awk',
                              '"{ print $2 }"'], stdout=subprocess.PIPE)
        device_names = ret.stdout.decode().strip().splitlines()
        cls.num_devices = len(device_names)
        all_devices = []
        for dev_idx in range(cls.num_devices) :
            details = subprocess.run(['ctpu', 'status', '-details', '-name', name],
                                     stdout=subprocess.PIPE)
            rx_hw_location = re.compile(r'^Compute Engine Machine Type:(.+)$')
            m = rx_hw_location.search(details)
            if m is not None:
                hw_location = m.group(1).strip()
            else:
                hw_location = 'unknown'
            # Memory size for TPU? For now, just set fixed value (compute engine's
            # memory).
            memory_size = 7.5 * (2 * 10) * (2 * 10) * (2 * 10)
            dev_info = TPUAcceleratorInfo(
                device_id=dev_idx,
                hw_location=f'{hw_location}@{device_names[dev_idx]}',
                memory_size=memory_size,
                processing_units=1,  # TPU sharing is not possible for now
            )
            all_devices.append(dev_info)
        return all_devices

    @classmethod
    async def generate_docker_args(cls, docker, proc_shares):
        tpus = []
        for dev_idx in range(cls.num_devices):
            if dev_idx not in proc_shares:
                tpus.append(dev_idx)
        return {
            # TODO: There's no way to use multiple tpus now. Only the first TPU will
            # be used.
            'Env': [
                f"TPU_VISIBLE_DEVICES={','.join(map(str, tpus))}",
                f"TPU_NAME={os.environ.get('TPU_NAME', '')}",
            ],
        }
