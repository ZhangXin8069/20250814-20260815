from typing import List, Union, Dict, Any, Type, Callable, NamedTuple, Literal
import logging
from sys import stdout
import os
from os import environ
from mpi4py import MPI


class _MPILogger:
    def __init__(self, root: int = 0) -> None:
        self.root = root
        formatter = logging.Formatter(fmt="{name} {levelname}: {message}", style="{")
        stdout_handler = logging.StreamHandler(stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        self.logger = logging.getLogger("mpi init")
        self.logger.level = logging.DEBUG
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Type[Warning]):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg), stack_info=True)

    def error(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


class _ComputeCapability(NamedTuple):
    major: int
    minor: int

_MPI_LOGGER: _MPILogger = _MPILogger()
_MPI_COMM: MPI.Intracomm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_SIZE: Union[List[int], None] = None
_GRID_COORD: Union[List[int], None] = None
_GRID_MAP: Literal["XYZT_FASTEST", "TZYX_FASTEST"] = "XYZT_FASTEST"
"""For MPI, the default node mapping is lexicographical with t varying fastest."""
_BACKEND: Literal["numpy", "cupy", "torch"] = "cupy"
_CUDA_IS_HIP: bool = False
_CUDA_DEVICE: int = -1
_CUDA_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def getRankFromCoord(grid_coord: List[int], grid_size: List[int] = None) -> int:
    def _process(xyzt: List[int]):
        if _GRID_MAP == "XYZT_FASTEST":
            return xyzt
        elif _GRID_MAP == "TZYX_FASTEST":
            return xyzt[::-1]
        else:
            _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)

    grid_size = _process(_GRID_SIZE if grid_size is None else grid_size)
    grid_coord = _process(grid_coord)
    mpi_rank = 0
    for g, G in zip(grid_coord, grid_size):
        mpi_rank = mpi_rank * G + g
    return mpi_rank


def getCoordFromRank(mpi_rank: int, grid_size: List[int] = None) -> List[int]:
    def _process(xyzt: List[int]):
        if _GRID_MAP == "XYZT_FASTEST":
            return xyzt[::-1]
        elif _GRID_MAP == "TZYX_FASTEST":
            return xyzt
        else:
            _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)

    if grid_size is None:
        assert isGridInitialized()
        grid_size = _process(_GRID_SIZE)
    else:
        grid_size = _process(grid_size)
    grid_coord = []
    for G in grid_size:
        grid_coord.append(mpi_rank % G)
        mpi_rank //= G
    grid_coord = _process(grid_coord)
    return grid_coord


def _composition(n: int, d: int):
    """
    Writing n as the sum of d natural numbers
    """
    addend: List[List[int]] = []
    i = [0 for _ in range(d - 1)] + [n] + [0]
    while i[0] <= n:
        addend.append([i[s] - i[s - 1] for s in range(d)])
        i[d - 2] += 1
        for s in range(d - 2, 0, -1):
            if i[s] == n + 1:
                i[s] = 0
                i[s - 1] += 1
        for s in range(1, d - 1, 1):
            if i[s] < i[s - 1]:
                i[s] = i[s - 1]
    return addend


def _factorization(k: int, d: int):
    """
    Writing k as the product of d positive numbers
    """
    prime_factor: List[List[List[int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([[p**a for a in addend] for addend in _composition(n, d)])
    if k != 1:
        prime_factor.append([[k**a for a in addend] for addend in _composition(1, d)])
    return prime_factor


def _partition(factor: List[List[List[int]]], sublatt_size: List[int], grid_size: List[int] = None, idx: int = 0):
    if idx == 0:
        grid_size = [1 for _ in range(len(sublatt_size))]
        factor = _factorization(factor, len(sublatt_size))
    if idx == len(factor):
        yield grid_size
    else:
        for factor_size in factor[idx]:
            for L, x in zip(sublatt_size, factor_size):
                if L % x != 0:
                    break
            else:
                yield from _partition(
                    factor,
                    [L // f for L, f in zip(sublatt_size, factor_size)],
                    [G * f for G, f in zip(grid_size, factor_size)],
                    idx + 1,
                )


def getDefaultGrid(mpi_size: int, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0, "lattice volume must be divisible by MPI size"
    
    evenodd = False
    
    if evenodd:
        assert (
            Lx % 2 == 0 and Ly % 2 == 0 and Lz % 2 == 0 and Lt % 2 == 0
        ), "lattice size must be even in all directions for even-odd preconditioning"
        partition = _partition(mpi_size, [Lx // 2, Ly // 2, Lz // 2, Lt // 2])
    else:
        partition = _partition(mpi_size, [Lx, Ly, Lz, Lt])
    for grid_size in partition:
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    if min_grid == []:
        _MPI_LOGGER.critical(
            f"Cannot get the proper grid for lattice size {latt_size} with {mpi_size} MPI processes", ValueError
        )
    return min(min_grid)

def initGrid(grid_size: List[int], latt_size: List[int] = None):
    global _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        if grid_size is None and latt_size is not None:
            grid_size = getDefaultGrid(_MPI_SIZE, latt_size)
        grid_size = grid_size if grid_size is not None else [1, 1, 1, 1]

        Gx, Gy, Gz, Gt = grid_size
        if _MPI_SIZE != Gx * Gy * Gz * Gt:
            _MPI_LOGGER.critical(f"The MPI size {_MPI_SIZE} does not match the grid size {grid_size}", ValueError)
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        _MPI_LOGGER.info(f"Using the grid size {_GRID_SIZE}")
    else:
        _MPI_LOGGER.warning("Grid is already initialized", RuntimeWarning)


def initDevice(
    backend: Literal["numpy", "cupy", "torch"] = None, 
    device: int = -1, 
    enable_mps: bool = False,
    cuda_device_count: int = None  # 允许手动指定numpy后端的设备数量
):
    global _BACKEND, _CUDA_IS_HIP, _CUDA_DEVICE, _CUDA_COMPUTE_CAPABILITY
    if _CUDA_DEVICE < 0:
        from platform import node as gethostname

        if backend is None:
            backend = environ.get("_BACKEND", "cupy")
        
        # 为numpy后端定义默认的模拟函数
        if backend == "numpy":
            # 默认设备数量：如果是CPU模拟，我们可以认为有无限多的"设备"
            # 或者通过cuda_device_count参数指定
            default_device_count = 0x7FFFFFFF if cuda_device_count is None else cuda_device_count
            
            def cudaGetDeviceCount() -> int:
                return default_device_count
            
            def cudaGetDeviceProperties(device: int) -> Dict[str, Any]:
                # 返回完整的设备属性，模拟GPU设备
                return {
                    "major": 0, 
                    "minor": 0,
                    "name": f"numpy-cpu-simulated-device-{device}",
                    "totalGlobalMem": 0,
                    "multiProcessorCount": 0,
                    "maxThreadsPerBlock": 0,
                    "maxThreadsDim": [0, 0, 0],
                    "maxGridSize": [0, 0, 0],
                    "sharedMemPerBlock": 0,
                    "totalConstMem": 0,
                    "warpSize": 0,
                    "memoryClockRate": 0,
                    "memoryBusWidth": 0,
                    "regsPerBlock": 0,
                    "clockRate": 0
                }
            
            def cudaSetDevice(device: int) -> None:
                # numpy后端不需要设置设备，但可以记录日志
                _MPI_LOGGER.info(f"numpy backend: using simulated device {device}")
                pass
            
            _CUDA_IS_HIP = False  # numpy后端不支持HIP
            
        elif backend == "cupy":
            import cupy
            from cupy.cuda.runtime import getDeviceCount as cudaGetDeviceCount
            from cupy.cuda.runtime import getDeviceProperties as cudaGetDeviceProperties
            from cupy.cuda.runtime import is_hip

            def cudaSetDevice(device: int) -> None:
                cupy.cuda.Device(device).use()
            
            _CUDA_IS_HIP = is_hip
            
        elif backend == "torch":
            import torch
            from torch.cuda import device_count as cudaGetDeviceCount
            from torch.cuda import get_device_properties as cudaGetDeviceProperties
            
            def cudaSetDevice(device: int) -> None:
                torch.cuda.set_device(device)
            
            # 检查是否是HIP
            try:
                from torch.version import hip
                _CUDA_IS_HIP = hip is not None
            except ImportError:
                _CUDA_IS_HIP = False
        else:
            _MPI_LOGGER.critical(f"Unsupported CUDA backend {backend}", ValueError)
        
        _BACKEND = backend
        _MPI_LOGGER.info(f"Using backend {backend}")
        
        # numpy后端特殊处理：不需要基于主机名的设备分配
        if backend == "numpy":
            # 对于numpy后端，我们只需要设置一个虚拟设备
            if device < 0:
                _CUDA_DEVICE = 0
            else:
                _CUDA_DEVICE = device
            
            # 设置模拟的计算能力
            _CUDA_COMPUTE_CAPABILITY = _ComputeCapability(0, 0)
            
            # 调用设备设置函数（虽然对numpy是空操作）
            cudaSetDevice(_CUDA_DEVICE)
            
            _MPI_LOGGER.info(f"numpy backend initialized with simulated device {_CUDA_DEVICE}")
            return  # numpy后端初始化完成，跳过GPU特有的逻辑
        
        # GPU后端（cupy/torch）的初始化逻辑
        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if device < 0:
            device_count = cudaGetDeviceCount()
            if device_count == 0:
                _MPI_LOGGER.critical("No devices found", RuntimeError)

            # We initialize gpuid if it's still negative.
            device = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    device += 1

            if device >= device_count:
                if enable_mps:
                    device %= device_count
                    # 使用日志而不是print，避免竞争条件
                    _MPI_LOGGER.info(f"MPS enabled, rank={_MPI_RANK:3d} -> gpu={device}")
                else:
                    _MPI_LOGGER.critical(f"Too few GPUs available on {hostname}", RuntimeError)
        _CUDA_DEVICE = device

        props = cudaGetDeviceProperties(device)
        
        # 统一设备属性访问方式
        if hasattr(props, "major") and hasattr(props, "minor"):
            major = int(props.major)
            minor = int(props.minor)
        else:
            # 可能是字典或命名元组
            major = int(props.get("major", 0) if isinstance(props, dict) else getattr(props, "major", 0))
            minor = int(props.get("minor", 0) if isinstance(props, dict) else getattr(props, "minor", 0))
        
        _CUDA_COMPUTE_CAPABILITY = _ComputeCapability(major, minor)

        cudaSetDevice(device)
        _MPI_LOGGER.info(f"{backend} backend initialized with device {device} (compute capability {major}.{minor})")
    else:
        _MPI_LOGGER.warning("Device is already initialized", RuntimeWarning)

def mpinit(
    grid_size: List[int], 
    latt_size: List[int] = None,
    backend: Literal["numpy", "cupy", "torch"] = None, 
    device: int = -1, 
    enable_mps: bool = False,
    cuda_device_count: int = None  # 允许手动指定numpy后端的设备数量
    
):
    initDevice(backend = backend, device = device, enable_mps = enable_mps, cuda_device_count = cuda_device_count)
    initGrid(grid_size, latt_size = latt_size)
    
def isGridInitialized():
    return _GRID_SIZE is not None


def isDeviceInitialized():
    return _CUDA_DEVICE >= 0


def getLogger():
    return _MPI_LOGGER


def setLoggerLevel(level: Literal["debug", "info", "warning", "error", "critical"]):
    _MPI_LOGGER.logger.setLevel(level.upper())


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize():
    if _GRID_SIZE is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return _GRID_SIZE


def getGridCoord():
    if _GRID_COORD is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return _GRID_COORD


def getGridMap():
    return _GRID_MAP


def setGridMap(grid_map: Literal["XYZT_FASTEST", "TZYX_FASTEST"]):
    global _GRID_MAP
    _GRID_MAP = grid_map


def getCUDABackend():
    return _BACKEND


def isHIP():
    return _CUDA_IS_HIP


def getCUDADevice():
    return _CUDA_DEVICE


def getCUDAComputeCapability():
    return _CUDA_COMPUTE_CAPABILITY


def isCUDABackend():
    """检查当前后端是否支持CUDA（cupy或torch）"""
    return _BACKEND in ["cupy", "torch"]


def isNumpyBackend():
    """检查当前是否为numpy后端"""
    return _BACKEND == "numpy"

def get_mpi_tlist(Nt, t, gtype:Literal['find', 'TScatter'] = 'find'):
    '''
    param:
        Nt:     the time of lattice space
        t:      the range t need to be scatter or find
        gtype:  scatter-> allocate time(t) to different rank
                find-> find time(t) in which rank and indx
                
    return:
        gtype == find:    1.rank and 2.indx of t
        gtype == TScatter or Scatter: 1.the range(t) scatter in different rank, 2.rank, 3.the indx of time in each rank.
         
    '''
    
    size = getMPISize()
    rank = getMPIRank()

    if gtype == 'find':
        if type(t) == int or type(t) == float:
            t_rank = int(t)%size
            t_rank_indx = ((int(t) + Nt)%Nt)//size
            
            return t_rank, t_rank_indx
        
        elif type(t) == list or type(t) == range:   
            t_list = [x%Nt for x in t]   
            list_rank = [x%size for x in t_list]
            t_list_rank_indx = [((x + Nt)%Nt)//size for x in t_list]

            return list_rank, t_list_rank_indx
        
        elif type(t).__module__ == 'numpy' or type(t).__module__ == 'cupy':
            t_list = [x%Nt for x in t.reshape(-1)]   
            list_rank = [x%size for x in t_list]
            t_list_rank_indx = [((x + Nt)%Nt)//size for x in t_list]
        
            return list_rank, t_list_rank_indx
        
        else:
            return None

    elif gtype == 'TScatter' or gtype == 'Scatter':
        if type(t) == int or type(t) == float:
            t_list = [t%Nt]

        elif type(t) == list or type(t) == range:
            t_list = [x%Nt for x in t]

        elif type(t).__module__ == 'numpy' or type(t).__module__ == 'cupy':
            t_list = [x%Nt for x in t.reshape(-1)]
        
        t_list_rank = [(x)%Nt for x in t_list if (x)%size == rank]
        list_rank = [rank for x in t_list if (x)%size == rank]

        t_list_rank_indx = [((x - rank + Nt)%Nt)//size for x in t_list_rank]
        return t_list_rank, list_rank, t_list_rank_indx
        
global mpi_send_data_tag
mpi_send_data_tag = 0

def get_mpi_data(
        data, 
        mdtype:Literal['Send', 'Gather', 'TGather', 'Allgather', 'Bcast', 'Scatter', 'TScatter', 'Transport'] = 'Gather', 
        root:int = 0, 
        recv_rank:int = 0,
        recv_buff = None,
        axis:int = 0
        ):
    
    '''
    Convenient customizable MPI strategy
    Args:
        data: the data need to be transmission
        mdtype: like mpi.comm....
        root: the data in whick rank
        recv_rank: if the transmission is point to piont this is the rank for receiving data is required
        recv_buff: not to be set
        axis: which axis like numpy
        
    Return:
        the transmitted data
    '''
    
    import numpy as np
    comm = getMPIComm()
    
    comm.Barrier()
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size == 1:
        return data.copy()
    
    if 'cupy' in str(type(data)):
        data = data.get()
        data_cupy = True

    else:
        data_cupy = False
        
    try:
        data = data.copy()

    except:
        if rank == root:
            data = data.copy()

    if mdtype == 'Send':
        if recv_rank != root:
            global mpi_send_data_tag
            mpi_send_data_tag = (mpi_send_data_tag + 3)%30000

            tag = mpi_send_data_tag

            if recv_rank >= size:
                print(f'recv_rank {recv_rank} must smaller then size {size}')
                comm.Abort(1)

            if rank == root:
                data = np.ascontiguousarray(data)
                comm.send(data.shape, dest = recv_rank, tag = tag + 1)
                comm.send(data.dtype, dest = recv_rank, tag = tag + 2)
                comm.Send(data, dest = recv_rank, tag = tag)

            elif rank == recv_rank:
                data_shape = comm.recv(source = root, tag = tag + 1)
                data_type = comm.recv(source = root, tag = tag + 2)

                if recv_buff is not None:
                    data_recv = recv_buff

                else:
                    data_recv = np.empty(data_shape, dtype = data_type)

                comm.Recv(data_recv, source = root, tag = tag)

            if rank == recv_rank:
                if data_cupy:
                    import cupy
                    data_recv = cupy.asarray(data_recv)

                return data_recv
            
            else:
                return None

        else:
            if rank == root:
                if data_cupy:
                    import cupy
                    data = cupy.asarray(data)

                return data
                
            else:
                return None
            
    elif mdtype == 'Gather':
        # data = np.moveaxis(data, source = axis, destination = 0)
        data_shape = [size] + list(data.shape)

        if rank == root:
            if recv_buff is not None:
                data_recv = recv_buff

            else:
                data_recv = np.empty(data_shape, dtype = data.dtype)

        else:
            data_recv = None
        
        data = np.ascontiguousarray(data)

        comm.Gather(data, data_recv, root = root)

        # if rank == root:
            # data_recv = np.moveaxis(data_recv, source = 0, destination = axis)
        
        if rank == root:
            if data_cupy:
                import cupy
                data_recv = cupy.asarray(data_recv)

            return data_recv
        
        else:
            return None
        
    elif mdtype == 'TGather':
        data = np.moveaxis(data, source = axis, destination = 0)
        data_shape = list(data.shape)
        data_shape = [size] + data_shape

        if rank == root:
            if recv_buff is not None:
                data_recv = recv_buff

            else:
                data_recv = np.empty(data_shape, dtype = data.dtype)

        else:
            data_recv = None
        
        data = np.ascontiguousarray(data)

        comm.Gather(data, data_recv, root = root)

        if rank == root:
            data_recv = np.moveaxis(data_recv, source = 0, destination = 1)
            data_recv = data_recv.reshape(tuple([size * data_shape[1]] + data_shape[2:]))

            data_recv = np.moveaxis(data_recv, source = 0, destination = axis)
        
        if rank == root:
            if data_cupy:
                import cupy
                data_recv = cupy.asarray(data_recv)

            return data_recv
        
        else:
            return None
            
    elif mdtype == 'Allgather':
        # data = np.moveaxis(data, source = axis, destination = 0)
        
        data_shape = list(data.shape)
        data_shape = [size] + data_shape

        if recv_buff is not None:
            data_recv = recv_buff

        else:
            data_recv = np.empty(data_shape, dtype = data.dtype)

        data = np.ascontiguousarray(data)

        comm.Allgather(data, data_recv)
        # if axis >=0:
        #     data_recv = np.moveaxis(data_recv, source = 1, destination = axis + 1)

        # else:
        #     data_recv = np.moveaxis(data_recv, source = 1, destination = axis)

        if data_cupy:
            import cupy
            data_recv = cupy.asarray(data_recv)

        return data_recv
    
    elif mdtype == 'Bcast':
        if rank == root:
            data = np.asarray(data)
            data_shape = list(data.shape)
            data_type = data.dtype

        else:
            data_shape = None
            data_type = None

        data_shape = comm.bcast(data_shape, root = root)
        data_type = comm.bcast(data_type, root = root)
        if recv_buff is not None:
            data_recv = recv_buff

        else:
            data_recv = np.empty(data_shape, dtype = data_type)

        if rank == root:
            data_recv = data
            data_recv = np.ascontiguousarray(data_recv)

        comm.Bcast(data_recv, root = root)

        if data_cupy:
            import cupy
            data_recv = cupy.asarray(data_recv)

        return data_recv

    elif mdtype == 'Scatter':
        if rank == root:
            data = np.moveaxis(data, source = axis, destination = 0)
            data = data.reshape([size, data.shape[0] // size] + list(data.shape)[1:])
            data_shape = list(data.shape)[1:]
            data_type = data.dtype
            data = np.ascontiguousarray(data)

        else:
            data_shape = None
            data_type = None

        data_shape = comm.bcast(data_shape, root = root)
        data_type = comm.bcast(data_type, root = root)
        if recv_buff is not None:
            data_recv = recv_buff

        else:
            data_recv = np.empty(data_shape, dtype = data_type)

        comm.Scatter(data, data_recv, root = root)
        data_recv = np.moveaxis(data_recv, source = 0, destination = axis)

        if data_cupy:
            import cupy
            data_recv = cupy.asarray(data_recv)

        return data_recv
    
    elif mdtype == 'TScatter':
        if rank == root:
            data = np.moveaxis(data, source = axis, destination = 0)
            remaind_size = data.shape[0]%size

            if remaind_size != 0:
                data_remaind = data[(data.shape[0]//size * size):]
                data = data[:(data.shape[0]//size * size)]

            data = np.moveaxis(data.reshape([data.shape[0] // size, size] + list(data.shape)[1:]), source = 1, destination = 0)
            data_shape = list(data.shape)[1:]
            data_type = data.dtype
            data = np.ascontiguousarray(data)
            
        else:
            remaind_size = None
            data_shape = None
            data_type = None

        remaind_size = comm.bcast(remaind_size, root = root)
        data_shape = comm.bcast(data_shape, root = root)
        data_type = comm.bcast(data_type, root = root)

        if recv_buff is not None and remaind_size == 0:
            data_recv = recv_buff

        else:
            data_recv = np.empty(data_shape, dtype = data_type)
        
        comm.Scatter(data, data_recv, root = root)

        for i in range(remaind_size):

            indx = i%size
            if rank == root:
                data_send_remaind = np.asarray([data_remaind[i]])
                data_send_remaind = np.ascontiguousarray(data_send_remaind)
            else:
                data_send_remaind = None

            data_recv_remaind =  get_mpi_data(data = data_send_remaind, mdtype = 'Send', root = root, recv_rank = indx, axis = 0)
            
            if rank == indx:
                data_recv = np.append(data_recv, data_recv_remaind, axis = 0)

        data_recv = np.moveaxis(data_recv, source = 0, destination = axis)

        if data_cupy:
            import cupy
            data_recv = cupy.asarray(data_recv)
            
        return data_recv

    elif mdtype == 'Transport':
        data = np.ascontiguousarray(data)
        data_recv = get_mpi_data(data = data, mdtype = 'Gather', root = root, axis = axis)

        data_recv = np.ascontiguousarray(data_recv)
        data_recv = get_mpi_data(data = data_recv, mdtype = 'TScatter', root = root, axis = axis)

        if data_cupy:
            import cupy
            data_recv = cupy.asarray(data_recv)

        return data_recv