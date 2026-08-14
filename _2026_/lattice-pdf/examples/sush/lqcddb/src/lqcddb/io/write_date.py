import numpy as np
import os
from typing import Literal,List, Tuple, Any
import itertools

def readin_eigvecs(file_path, Nx):
    f=open(file_path, 'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    
    return eigvecs

def readin_peram(peram_dir, conf_id, Nt, Nev1): # One less parameter compared to pervious verison
    peram_cpu_all=np.zeros((Nt,Nt,4,4,Nev1,Nev1),dtype=complex) #t_source, t_sink, d_sink, d_source, ev_sink, ev_source,  complex
    for t_source in range(0,Nt):
        f=open("%s/perams.%s.0.%i"%(peram_dir,conf_id,t_source),'rb')
        peram=np.fromfile(f,dtype='f8')
        f.close()
        
        for d_source in range(1,4):
            f=open("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source),'rb')
            temp=np.fromfile(f,dtype='f8')
            peram=np.append(peram, temp)
            temp=None
            f.close()
        peram_size=peram.size
        Nev=int(np.sqrt(peram_size/(4*4*Nt*2)))
        peram=peram.reshape(4,Nt,Nev,4,Nev,2) #d_source, t_sink, ev_source, d_sink, ev_sink, complex
        peram=peram.transpose(1,3,0,4,2,5) #t_sink, d_sink, d_source, ev_sink, ev_source,  complex
        peram=peram[...,0] + peram[...,1]*1j
        peram_cpu_all[t_source]=peram[:,:,:,0:Nev1,0:Nev1]
    
    return peram_cpu_all

def write_data_ascii( data, T, L, filename, complex=True, verbose=False):
    """Writes the data into a file.

    The file is written to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.

    Args:
        filename: The filename of the file.
        data: The numpy array with data.
        verbose: The amount of info shown.
    """
    # check file
    check_write(filename)
    if verbose:
        print("saving to file " + str(filename))

    # in case the dimension is 1, treat the data as one sample
    # to make the rest easier we add an extra axis
    nsamples = data.shape[0]
#    T = data.shape[1]
#    L = int(T/2) 
    _data = data.reshape((T*nsamples), -1)
    _counter = np.fromfunction(lambda i, *j: i%T,
                            (_data.shape[0],) + (1,)*(len(_data.shape)-1),
                            dtype=int)
    if complex:
        head = "%i %i %i %i %i" % (nsamples, T, 1, L, 1)
        data_real = _data.real
        data_imag = _data.imag
        _fdata = np.concatenate((_counter, data_real, data_imag), axis=1)
        savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.32f", "%.32f"])
    else:
        head = "%i %i %i %i %i" % (nsamples, T, 0, L, 1)
        _fdata = np.concatenate((_counter, _data), axis=1)
        savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.32f"])

# ------------------------------------------------------------------------------------------------
def check_write(filename):
    """Do some checks before writing a file.
    """
    # check if path exists, if not then create it
    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")

# ------------------------------------------------------------------------------------------------
def savetxt( fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
                footer='', comments='# '):
    """This code is from NumPy 1.9.1. For help see there.

    It was included because features are used that were added in version 1.7
    but on some machines only NumPy version 1.6.2 is available.
    """
    ## needed for the rest
    from numpy.compat import asstr, asbytes

    def _is_string_like(obj):
        try:
            obj + ''
        except (TypeError, ValueError):
            return False
        return True

    # Py3 conversions first
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
        delimiter = asstr(delimiter)

    own_fh = False
    if _is_string_like(fname):
        own_fh = True
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, 'wb')
        else:
            if os.sys.version_info[0] >= 3:
                fh = open(fname, 'wb')
            else:
                fh = open(fname, 'w')
    elif hasattr(fname, 'write'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.descr)
        else:
            ncol = X.shape[1]

        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '%10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = asstr(delimiter).join(map(asstr, fmt))
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))

        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + header + newline))
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.append(number.real)
                    row2.append(number.imag)
                fh.write(asbytes(format % tuple(row2) + newline))
        else:
            for row in X:
                fh.write(asbytes(format % tuple(row) + newline))
        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + footer + newline))
    finally:
        if own_fh:
            fh.close()

def check_dir_path(save_path):
    import pathlib

    path = pathlib.Path(save_path)

    if path.exists():
        print('save_path:',save_path)

    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',save_path)

def check_files_existence(path_templates: List[str], **kwargs) -> Tuple[List[Any], List[Any]]:
    """
    检查任意数量占位符替换后，所有模板文件是否都存在，且每种文件大小一致。

    除了确保文件存在外，还对每种模板文件进行大小一致性检查：
    以第一个存在的组合的文件大小作为基准，后续组合中任何同种文件大小与基准不一致
    的视为存储错误，归入 missing。

    参数:
        path_templates: 包含占位符的文件路径模板列表。
                        占位符格式为 '<name>'，例如 '<exp>/<run>/file'。
        **kwargs:       占位符名称到取值列表的映射。
                        例如 exp=['E1','E2'], run=[1,2]
                        表示需要检查 <exp> 和 <run> 所有组合对应的文件。

    返回:
        (existing, missing) 两个列表。
        - 如果只传入一个占位符，列表中直接存放该占位符的取值（保持简洁）；
        - 如果传入多个占位符，列表中存放字典，键为占位符名，值为对应取值。
    """
    
    if not kwargs:
        raise ValueError("至少需要提供一个占位符参数，例如 run_id=[...]")

    # 提取占位符名称和对应的值列表
    placeholders = list(kwargs.keys())
    value_lists = [kwargs[p] for p in placeholders]

    existing = []          # 文件正常且存在
    missing = []           # 文件不存在
    corrupted = []         # 文件存在但大小异常

    # 辅助函数：构造 mapping 并解析所有文件路径
    def _resolve_paths(combo):
        """根据 combo 解析出所有模板对应的实际文件路径列表"""
        if len(placeholders) == 1:
            mapping = {f"<{p}>": str(combo) for p in placeholders}
        else:
            mapping = {f"<{p}>": str(combo[p]) for p in placeholders}
        paths = []
        for template in path_templates:
            fp = template
            for tag, val in mapping.items():
                fp = fp.replace(tag, val)
            paths.append(fp)
        return paths

    # 辅助函数：构造 combo 对象
    def _make_combo(values):
        if len(placeholders) == 1:
            return values[0]
        else:
            return dict(zip(placeholders, values))

    # 基准文件大小（由第一个全部文件存在的组合确定）
    reference_sizes: List[int] = []

    # 遍历所有取值的笛卡尔积
    for values in itertools.product(*value_lists):
        combo = _make_combo(values)
        paths = _resolve_paths(combo)

        # 检查所有文件是否存在
        all_exist = all(os.path.exists(fp) for fp in paths)
        if not all_exist:
            missing.append(combo)
            continue

        # 如果尚未设定基准大小，以第一个存在的组合为准
        if not reference_sizes:
            reference_sizes = [os.path.getsize(fp) for fp in paths]
            existing.append(combo)
        else:
            # 逐一比较每种文件的大小是否与基准一致
            size_ok = True
            for i, fp in enumerate(paths):
                if os.path.getsize(fp) != reference_sizes[i]:
                    size_ok = False
                    break
            if size_ok:
                existing.append(combo)
            else:
                corrupted.append(combo)

    # 汇总输出
    print(f"文件正常且存在 (len={len(existing)}): {existing}")
    print(f"文件异常但存在 (len={len(corrupted)}): {corrupted}")
    print(f"文件不存在 (len={len(missing)}): {missing}")

    return existing, missing + corrupted

def safe_save(file, arr, allow_pickle=True, fix_imports=True, fallback_dirs=None):
    """将数组保存为 NumPy ``.npy`` 二进制文件，出错时自动回退到备用目录。

    用法与 ``numpy.save`` 一致。先尝试保存到主路径；若失败（如磁盘满、
    权限不足），依次尝试 ``fallback_dirs``（用户指定）；若仍然失败，
    自动构建 ``/nexdata/project/lqcd/${USER}/result/`` 下的路径再试。

    自动回退路径的构建规则：
    从 ``file`` 中找到名称包含 ``result`` 的目录（如
    ``/public/group/lqcd/result/.../a.npy`` 或
    ``/public/group/lqcd/result_gen/.../b.npy``），提取该目录之后的部分
    作为子路径，拼接到 ``/nexdata/project/lqcd/${USER}/result/`` 下。
    例如 ``/public/group/lqcd/result_gen/foo/bar/a.npy``
    → ``/nexdata/project/lqcd/${USER}/result/foo/bar/a.npy``。
    若路径中不含 ``result`` 目录，则直接使用原文件名。

    Parameters
    ----------
    file : str 或 pathlib.Path
        目标文件路径。若不以 ``.npy`` 结尾，自动追加 ``.npy`` 后缀。
        仅支持字符串/Path，不支持文件对象。
    arr : array_like
        待保存的数组数据。
    allow_pickle : bool, optional
        是否允许使用 pickle 保存对象数组。默认 True。
    fix_imports : bool, optional
        仅用于 Python 2 兼容。默认 True。
    fallback_dirs : list of str, optional
        用户指定的备用目录列表，优先级高于自动回退目录。

    Returns
    -------
    str
        实际保存成功的文件路径。

    Raises
    ------
    OSError
        当所有路径（主路径 + 用户备用 + 自动回退）都保存失败时抛出。
    """
    import os

    # ── 规范化文件路径 ──────────────────────────────────────────
    file = str(file)
    if not file.endswith('.npy'):
        file = file + '.npy'

    basename = os.path.basename(file)

    # ── 辅助：尝试保存到指定目录 ─────────────────────────────────
    def _try_save(dir_path):
        """尝试保存到 dir_path 目录下，成功返回完整路径，失败返回 None。"""
        candidate = os.path.join(dir_path, basename)
        try:
            check_dir_path(dir_path)
            # 若 arr 为 cupy 数组，先转为 numpy 再保存
            _arr = arr
            if hasattr(_arr, 'get') and callable(_arr.get):
                _arr = _arr.get()
            np.save(candidate, _arr, allow_pickle=allow_pickle, fix_imports=fix_imports)
            return candidate
        except OSError as e:
            print(f"safe_save: failed to save to '{candidate}': {e}")
            return None

    # ── 1. 尝试主路径 ───────────────────────────────────────────
    primary_dir = os.path.dirname(file) or '.'
    result = _try_save(primary_dir)
    if result is not None:
        return result

    # ── 2. 尝试用户指定的备用目录 ─────────────────────────────────
    if fallback_dirs:
        for d in fallback_dirs:
            result = _try_save(d)
            if result is not None:
                print(f"safe_save: saved to fallback path '{result}'")
                return result

    # ── 3. 构建自动回退路径 /nexdata/project/lqcd/${USER}/result/ ─
    _file_abs = os.path.abspath(file)
    _parts = _file_abs.split(os.sep)
    _result_idx = None
    for _i, _p in enumerate(_parts):
        if 'result' in _p:
            _result_idx = _i
            break

    if _result_idx is not None and _result_idx + 1 < len(_parts):
        # result* 目录之后的部分（含文件名）
        _suffix = os.sep.join(_parts[_result_idx + 1:])
    else:
        _suffix = basename

    _user = os.environ.get('USER', '')
    if _user:
        _auto_fallback_dir = os.path.join(
            '/nexdata/project/lqcd', _user, 'result',
            os.path.dirname(_suffix)
        )
        result = _try_save(_auto_fallback_dir)
        if result is not None:
            print(f"safe_save: saved to auto fallback path '{result}'")
            return result

    # ── 4. 全部失败 ─────────────────────────────────────────────
    raise OSError(
        f"safe_save: all save locations exhausted for '{basename}'"
    )

