import numpy as np
import os
from typing import Literal
from .corr_base_functions import ArraySlicer
def load(Nx, path, dtype:Literal['numpy', 'cupy', 'pickle', 'readin'] = 'numpy', vec_or_peram: Literal['peram', 'vector'] = 'vector'):
    '''
    path 是读取文件的路径，不允许出现参量
    
    dtype 是读取的和读取后的数组类型 numpy 和 cupy 都可以读取 pickle 类型 其中除了 cupy 其余数组均为 numpy 数组

    vec_or_peram 是读取的文件是 vector or perambulator
    '''

    if dtype == 'numpy':
        import numpy as np
        return np.load(path, allow_pickle = True)
        
    elif dtype == 'cupy':
        import cupy as cp
        return cp.load(path, allow_pickle = True)
    
    elif dtype == 'pickle':
        import pickle 
        return pickle.load(open(path,'rb'))
    
    elif dtype == 'readin' and vec_or_peram == 'vector':
        return readin_eigvecs(path, Nx = Nx)
    
    elif dtype == 'readin' and vec_or_peram == 'peram':
        return readin_peram(path, Nx = Nx)

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

