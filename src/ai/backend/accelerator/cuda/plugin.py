import subprocess
import logging
from pathlib import Path
from pprint import pformat
import re
from typing import Collection, Mapping, Sequence

import attr
import aiohttp

from ai.backend.agent.resources import (
    AbstractComputeDevice, AbstractComputePlugin, AbstractAllocMap,
    DiscretePropertyAllocMap,
    get_resource_spec_from_container,
)
from ai.backend.agent.stats import (
    StatContext, NodeMeasurement, ContainerMeasurement
)
from ai.backend.common.logging import BraceStyleAdapter
from . import __version__
from .nvidia import libcudart

log = BraceStyleAdapter(logging.getLogger('ai.backend.accelerator.cuda'))


async def init(config: Mapping[str, str]):
    try:
        ret = subprocess.run(['nvidia-docker', 'version'],
                             stdout=subprocess.PIPE)
    except FileNotFoundError:
        log.warning('nvidia-docker is not installed.')
        log.info('CUDA acceleration is disabled.')
        CUDAPlugin.enabled = False
        return CUDAPlugin
    rx = re.compile(r'^NVIDIA Docker: (\d+\.\d+\.\d+)')
    for line in ret.stdout.decode().strip().splitlines():
        m = rx.search(line)
        if m is not None:
            CUDAPlugin.nvdocker_version = tuple(map(int, m.group(1).split('.')))
            break
    else:
        log.error('could not detect nvidia-docker version!')
        log.info('CUDA acceleration is disabled.')
        CUDAPlugin.enabled = False
        return CUDAPlugin
    device_mask = await config.get('device_mask')
    if device_mask is not None:
        CUDAPlugin.device_mask = [*device_mask.split(',')]
    try:
        detected_devices = await CUDAPlugin.list_devices()
        log.info('detected devices:\n' + pformat(detected_devices))
        log.info('nvidia-docker version: {}', CUDAPlugin.nvdocker_version)
        log.info('CUDA acceleration is enabled.')
    except ImportError:
        log.warning('could not load the CUDA runtime library.')
        log.info('CUDA acceleration is disabled.')
        CUDAPlugin.enabled = False
    except RuntimeError as e:
        log.warning('CUDA init error: {}', e)
        log.info('CUDA acceleration is disabled.')
        CUDAPlugin.enabled = False
    return CUDAPlugin


@attr.s(auto_attribs=True)
class CUDADevice(AbstractComputeDevice):
    pass


class CUDAPlugin(AbstractComputePlugin):

    key = 'cuda'
    slot_types = (
        ('cuda.device', 'count'),
    )

    device_mask = []
    enabled = True

    nvdocker_version = (0, 0, 0)

    @classmethod
    async def list_devices(cls) -> Collection[CUDADevice]:
        if not cls.enabled:
            return []
        all_devices = []
        num_devices = libcudart.get_device_count()
        for dev_id in map(str, range(num_devices)):
            if dev_id in cls.device_mask:
                continue
            raw_info = libcudart.get_device_props(int(dev_id))
            sysfs_node_path = "/sys/bus/pci/devices/" \
                              f"{raw_info['pciBusID_str'].lower()}/numa_node"
            try:
                node = int(Path(sysfs_node_path).read_text().strip())
            except OSError:
                node = None
            dev_info = CUDADevice(
                device_id=dev_id,
                hw_location=raw_info['pciBusID_str'],
                numa_node=node,
                memory_size=raw_info['totalGlobalMem'],
                processing_units=raw_info['multiProcessorCount'],
            )
            all_devices.append(dev_info)
        return all_devices

    @classmethod
    async def available_slots(cls) -> Mapping[str, str]:
        devices = await cls.list_devices()
        slots = {
            # TODO: move to physical device info reports
            # 'cuda.smp': sum(dev.processing_units for dev in devices),
            # 'cuda.mem': f'{BinarySize(sum(dev.memory_size for dev in devices)):g}',
            'cuda.device': len(devices),
        }
        return slots

    @classmethod
    def get_version(cls) -> str:
        return __version__

    @classmethod
    async def extra_info(cls) -> Mapping[str, str]:
        if cls.enabled:
            try:
                return {
                    'cuda_support': True,
                    'cuda_version': '{0[0]}.{0[1]}'.format(libcudart.get_version()),
                }
            except (RuntimeError, ImportError):
                cls.enabled = False
        return {
            'cuda_support': False,
        }

    @classmethod
    async def gather_node_measures(cls, ctx: StatContext) \
                                  -> Sequence[NodeMeasurement]:
        return []

    @classmethod
    async def gather_container_measures(cls, ctx: StatContext) \
                                       -> Sequence[ContainerMeasurement]:
        return []

    @classmethod
    async def create_alloc_map(cls) -> AbstractAllocMap:
        devices = await cls.list_devices()
        return DiscretePropertyAllocMap(
            devices=devices,
            prop_func=lambda dev: 1)

    @classmethod
    async def get_hooks(cls, distro: str, arch: str) -> Sequence[Path]:
        return []

    @classmethod
    async def generate_docker_args(cls, docker, device_alloc):
        if not cls.enabled:
            return {}
        active_device_ids = set()
        for slot_type, per_device_alloc in device_alloc.items():
            for dev_id, alloc in per_device_alloc.items():
                if alloc > 0:
                    active_device_ids.add(dev_id)
        if cls.nvdocker_version[0] == 1:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(raise_for_status=True,
                                             timeout=timeout) as sess:
                try:
                    nvdocker_url = 'http://localhost:3476/docker/cli/json'
                    async with sess.get(nvdocker_url) as resp:
                        nvidia_params = await resp.json()
                except aiohttp.ClientError:
                    raise RuntimeError('NVIDIA Docker plugin is not available.')

            volumes = await docker.volumes.list()
            existing_volumes = set(vol['Name'] for vol in volumes['Volumes'])
            required_volumes = set(vol.split(':')[0]
                                   for vol in nvidia_params['Volumes'])
            missing_volumes = required_volumes - existing_volumes
            binds = []
            for vol_name in missing_volumes:
                for vol_param in nvidia_params['Volumes']:
                    if vol_param.startswith(vol_name + ':'):
                        _, _, permission = vol_param.split(':')
                        driver = nvidia_params['VolumeDriver']
                        await docker.volumes.create({
                            'Name': vol_name,
                            'Driver': driver,
                        })
            for vol_name in required_volumes:
                for vol_param in nvidia_params['Volumes']:
                    if vol_param.startswith(vol_name + ':'):
                        _, mount_pt, permission = vol_param.split(':')
                        binds.append('{}:{}:{}'.format(
                            vol_name, mount_pt, permission))
            devices = []
            for dev in nvidia_params['Devices']:
                m = re.search(r'^/dev/nvidia(\d+)$', dev)
                if m is None:
                    # Always add non-GPU device files required by the driver.
                    # (e.g., nvidiactl, nvidia-uvm, ... etc.)
                    devices.append(dev)
                    continue
                dev_id = m.group(1)
                if dev_id not in active_device_ids:
                    continue
                devices.append(dev)
            devices = [{
                'PathOnHost': dev,
                'PathInContainer': dev,
                'CgroupPermissions': 'mrw',
            } for dev in devices]
            return {
                'HostConfig': {
                    'Binds': binds,
                    'Devices': devices,
                },
            }
        elif cls.nvdocker_version[0] == 2:
            device_list_str = ','.join(sorted(active_device_ids))
            return {
                'HostConfig': {
                    'Runtime': 'nvidia',
                },
                'Env': [
                    f"NVIDIA_VISIBLE_DEVICES={device_list_str}",
                ],
            }
        else:
            raise RuntimeError('BUG: should not be reached here!')

    @classmethod
    async def restore_from_container(cls, container, alloc_map):
        if not cls.enabled:
            return
        assert isinstance(alloc_map, DiscretePropertyAllocMap)
        resource_spec = await get_resource_spec_from_container(container)
        if resource_spec is None:
            return
        alloc_map.allocations['cuda.device'].update(
            resource_spec.allocations.get('cuda', {}).get('cuda.device', {}))
