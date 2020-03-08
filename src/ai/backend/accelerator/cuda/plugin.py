from decimal import Decimal
import subprocess
import logging
from pathlib import Path
from pprint import pformat
import re
from typing import (
    Any, Optional,
    Collection, Set,
    Mapping, MutableMapping,
    Sequence, List,
    Tuple,
)
import uuid

import attr
import aiohttp

from ai.backend.common.logging import BraceStyleAdapter
from ai.backend.agent.resources import (
    AbstractComputeDevice, AbstractComputePlugin,
    AbstractAllocMap, DiscretePropertyAllocMap,
)
try:
    from ai.backend.agent.resources import get_resource_spec_from_container  # type: ignore
except ImportError:
    from ai.backend.agent.docker.resources import get_resource_spec_from_container
from ai.backend.agent.stats import (
    StatContext, MetricTypes,
    NodeMeasurement, ContainerMeasurement, Measurement,
)
from ai.backend.agent.types import Container
from ai.backend.common.types import (
    BinarySize, MetricKey,
    DeviceName, DeviceId, DeviceModelInfo,
    SlotName, SlotTypes,
)
from . import __version__
from .nvidia import libcudart, libnvml, LibraryError

__all__ = (
    'PREFIX',
    'CUDADevice',
    'CUDAPlugin',
    'init',
)

PREFIX = 'cuda'

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
    raw_device_mask = config.get('device_mask')
    if raw_device_mask is not None:
        CUDAPlugin.device_mask = [
            *map(lambda dev_id: DeviceId(dev_id), raw_device_mask.split(','))
        ]
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
    model_name: str
    uuid: str


class CUDAPlugin(AbstractComputePlugin):

    key = DeviceName('cuda')
    slot_types: Sequence[Tuple[SlotName, SlotTypes]] = (
        (SlotName('cuda.device'), SlotTypes('count')),
    )

    device_mask: Sequence[DeviceId] = []
    enabled: bool = True

    nvdocker_version: Tuple[int, ...] = (0, 0, 0)

    @classmethod
    async def list_devices(cls) -> Collection[CUDADevice]:
        if not cls.enabled:
            return []
        all_devices = []
        num_devices = libcudart.get_device_count()
        for dev_id in map(lambda idx: DeviceId(str(idx)), range(num_devices)):
            if dev_id in cls.device_mask:
                continue
            raw_info = libcudart.get_device_props(int(dev_id))
            sysfs_node_path = "/sys/bus/pci/devices/" \
                              f"{raw_info['pciBusID_str'].lower()}/numa_node"
            node: Optional[int]
            try:
                node = int(Path(sysfs_node_path).read_text().strip())
            except OSError:
                node = None
            dev_uuid, raw_dev_uuid = None, raw_info.get('uuid', None)
            if raw_dev_uuid is not None:
                dev_uuid = str(uuid.UUID(bytes=raw_dev_uuid))
            else:
                dev_uuid = '00000000-0000-0000-0000-000000000000'
            dev_info = CUDADevice(
                device_id=dev_id,
                hw_location=raw_info['pciBusID_str'],
                numa_node=node,
                memory_size=raw_info['totalGlobalMem'],
                processing_units=raw_info['multiProcessorCount'],
                model_name=raw_info['name'],
                uuid=dev_uuid,
            )
            all_devices.append(dev_info)
        return all_devices

    @classmethod
    async def available_slots(cls) -> Mapping[SlotName, Decimal]:
        devices = await cls.list_devices()
        return {
            SlotName('cuda.device'): Decimal(len(devices)),
        }

    @classmethod
    def get_version(cls) -> str:
        return __version__

    @classmethod
    async def extra_info(cls) -> Mapping[str, Any]:
        if cls.enabled:
            try:
                return {
                    'cuda_support': True,
                    'nvidia_version': libnvml.get_driver_version(),
                    'cuda_version': '{0[0]}.{0[1]}'.format(libcudart.get_version()),
                }
            except ImportError:
                log.warning('extra_info(): NVML/CUDA runtime library is not found')
            except LibraryError as e:
                log.warning('extra_info(): {!r}', e)
        return {
            'cuda_support': False,
        }

    @classmethod
    async def gather_node_measures(
            cls, ctx: StatContext,
            ) -> Sequence[NodeMeasurement]:
        dev_count = 0
        mem_avail_total = 0
        mem_used_total = 0
        mem_stats = {}
        util_total = 0
        util_stats = {}
        if cls.enabled:
            try:
                dev_count = libnvml.get_device_count()
                for dev_id in map(lambda idx: DeviceId(str(idx)), range(dev_count)):
                    if dev_id in cls.device_mask:
                        continue
                    dev_stat = libnvml.get_device_stats(int(dev_id))
                    mem_avail_total += dev_stat.mem_total
                    mem_used_total += dev_stat.mem_used
                    mem_stats[dev_id] = Measurement(Decimal(dev_stat.mem_used),
                                                    Decimal(dev_stat.mem_total))
                    util_total += dev_stat.gpu_util
                    util_stats[dev_id] = Measurement(Decimal(dev_stat.gpu_util), Decimal(100))
            except ImportError:
                log.warning('gather_node_measure(): NVML library is not found')
            except LibraryError as e:
                log.warning('gather_node_measure(): {!r}', e)
        return [
            NodeMeasurement(
                MetricKey('cuda_mem'),
                MetricTypes.USAGE,
                unit_hint='bytes',
                stats_filter=frozenset({'max'}),
                per_node=Measurement(Decimal(mem_used_total), Decimal(mem_avail_total)),
                per_device=mem_stats,
            ),
            NodeMeasurement(
                MetricKey('cuda_util'),
                MetricTypes.USAGE,
                unit_hint='percent',
                stats_filter=frozenset({'avg', 'max'}),
                per_node=Measurement(Decimal(util_total), Decimal(dev_count * 100)),
                per_device=util_stats,
            ),
        ]

    @classmethod
    async def gather_container_measures(
            cls, ctx: StatContext,
            container_ids: Sequence[str],
            ) -> Sequence[ContainerMeasurement]:
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
    async def generate_docker_args(cls, docker,
                                   device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]]) \
                                   -> Mapping[str, Any]:
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
    async def get_attached_devices(
            cls, device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
            ) -> Sequence[DeviceModelInfo]:
        device_ids: List[DeviceId] = []
        if SlotName('cuda.devices') in device_alloc:
            device_ids.extend(device_alloc[SlotName('cuda.devices')].keys())
        available_devices = await cls.list_devices()
        attached_devices: List[DeviceModelInfo] = []
        for device in available_devices:
            if device.device_id in device_ids:
                proc = device.processing_units
                mem = BinarySize(device.memory_size)
                attached_devices.append({  # TODO: update common.types.DeviceModelInfo
                    'device_id': device.device_id,
                    'model_name': device.model_name,
                    'smp': proc,
                    'mem': mem,
                })
        return attached_devices

    @classmethod
    async def restore_from_container(
            cls, container: Container,
            alloc_map: AbstractAllocMap,
            ) -> None:
        if not cls.enabled:
            return
        resource_spec = await get_resource_spec_from_container(container.backend_obj)
        if resource_spec is None:
            return
        if hasattr(alloc_map, 'apply_allocation'):
            alloc_map.apply_allocation({
                SlotName('cuda.device'): resource_spec.allocations.get(
                    DeviceName('cuda'), {}
                ).get(
                    SlotName('cuda.device'), {}
                ),
            })
        else:
            alloc_map.allocations[SlotName('cuda.device')].update(
                resource_spec.allocations.get(
                    DeviceName('cuda'), {}
                ).get(
                    SlotName('cuda.device'), {}
                )
            )

    @classmethod
    async def generate_resource_data(
            cls, device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
            ) -> Mapping[str, str]:
        data: MutableMapping[str, str] = {}
        if not cls.enabled:
            return data

        active_device_id_set: Set[DeviceId] = set()
        for slot_type, per_device_alloc in device_alloc.items():
            for dev_id, alloc in per_device_alloc.items():
                if alloc > 0:
                    active_device_id_set.add(dev_id)
        active_device_ids = sorted(active_device_id_set, key=lambda v: int(v))
        data['CUDA_GLOBAL_DEVICE_IDS'] = ','.join(
            f'{local_idx}:{global_id}'
            for local_idx, global_id in enumerate(active_device_ids))
        return data
