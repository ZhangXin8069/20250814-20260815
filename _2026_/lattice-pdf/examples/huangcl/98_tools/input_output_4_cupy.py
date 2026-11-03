#!/beegfs/home/liuming/software/install/python/bin/python3
import os
import numpy as np
import cupy as cp
from opt_einsum import contract
# from gamma_matrix_cupy_DR import *

# ------------------------------------------------------------------------------------------------
def Peram_truncated(peram):
    peram_tc=peram[:,0:2,0:2,:,:]
    return peram_tc

# ------------------------------------------------------------------------------
def readin_eigvecs_device(eig_dir, t, Nev1,conf_id, Nx):
    f=open("%s/eigvecs_t%03d_%s"%(eig_dir, t, conf_id),'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    eigvecs=eigvecs[0:Nev1,:,:]
    eigvecs_cupy=cp.asarray(eigvecs)
    return eigvecs_cupy

# ------------------------------------------------------------------------------
def readin_eigvecs_cpu(eig_dir, t, Nev1,conf_id, Nx):
    f=open("%s/eigvecs_t%03d_%s"%(eig_dir, t, conf_id),'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    eigvecs=eigvecs[0:Nev1,:,:]
    # eigvecs_cupy=cp.asarray(eigvecs)
    return eigvecs

# ------------------------------------------------------------------------------------------------
# Based on "readin_peram_v2", read-in perambulator from one particular timeslice, and saved it in device memory
def readin_peram_device(peram_dir, conf_id, Nt, Nev1, t_source):
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
    peram=peram.transpose(1,3,0,4,2,5) #t_sink, d_sink, d_source, ev_sink, ev_souce,  complex
    peram=peram[...,0] + peram[...,1]*1j
    peram=peram[:,:,:,0:Nev1,0:Nev1]
    peram_cupy=cp.array(peram)
    return peram_cupy

# ------------------------------------------------------------------------------------------------
# Basically identical to v2, except the data is stored on RAM
def readin_peram_all_cpu(peram_dir, conf_id, Nt, Nev1): # One less parameter compared to pervious verison
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
    # peram_cupy=cp.array(peram)
    return peram_cpu_all

# ------------------------------------------------------------------------------------------------
def readin_VVV_device(VVV_dir, Nev1, t, conf_id, Px, Py, Pz):
    VVV=np.zeros((Nev1, Nev1, Nev1),dtype=complex)
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV=np.copy(temp)
    f.close()
    VVV_cupy=cp.array(VVV)
    
    return VVV_cupy

# ------------------------------------------------------------------------------------------------
def readin_VVV_cpu(VVV_dir, Nev1, t, conf_id, Px, Py, Pz):
    VVV=np.zeros((Nev1, Nev1, Nev1),dtype=complex)  
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV=np.copy(temp)
    f.close()
    # VVV_cupy=cp.array(VVV)
    return VVV

# ------------------------------------------------------------------------------------------------
def readin_VVV_all_device(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  VVV=np.zeros((Nt, Nev1, Nev1, Nev1),dtype=complex)
  for t in range(0,Nt):
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV[t]=np.copy(temp)
    f.close()
  VVV_cupy=cp.array(VVV)
  return VVV_cupy

# ------------------------------------------------------------------------------------------------
# Basically identical to original, except VVV is saved on the host, not on the device.
def readin_VVV_all_cpu(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  VVV=np.zeros((Nt, Nev1, Nev1, Nev1),dtype=complex)
  for t in range(0,Nt):
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV[t]=np.copy(temp)
    f.close()
  # VVV_cupy=cp.array(VVV)
  return VVV

# ------------------------------------------------------------------------------------------------
def readin_VdV_all_device(VdV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  f=open("%s/VdaggerV.Px%dPy%dPz%d.conf%s"%(VdV_dir, Px, Py, Pz, conf_id),'rb')
  VdV=np.fromfile(f,dtype='f8')
  Nev=int(np.sqrt(VdV.size/(Nt*2)))
  VdV=VdV.reshape(Nt, Nev, Nev, 2)
  VdV=VdV[...,0] + VdV[...,1]*1j
  VdV=VdV[:,0:Nev1, 0:Nev1]
  VdV_cupy=cp.array(VdV)
  f.close()
  return VdV_cupy	

# ------------------------------------------------------------------------------------------------
def readin_VdV_all_cpu(VdV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  f=open("%s/VdaggerV.Px%dPy%dPz%d.conf%s"%(VdV_dir, Px, Py, Pz, conf_id),'rb')
  VdV=np.fromfile(f,dtype='f8')
  Nev=int(np.sqrt(VdV.size/(Nt*2)))
  VdV=VdV.reshape(Nt, Nev, Nev, 2)
  VdV=VdV[...,0] + VdV[...,1]*1j
  VdV=VdV[:,0:Nev1, 0:Nev1]
  f.close()
#   VdV_cupy=cp.array(VdV)
  return VdV

# ------------------------------------------------------------------------------------------------
def write_data_ascii(data, T, L, filename, complex=True, verbose=False):
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
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
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

# calculate the 2pt exp part
def phase_exp_2pt(Mom,Nx):
  exp_diag = np.zeros(Nx*Nx*Nx*3, dtype=complex)   # M
  for z in range(0,Nx):
      for y in range(0,Nx):
          for x in range(0,Nx):
              Pos = np.array([z,y,x])
              exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3] = np.exp( -np.dot(Mom,Pos)*2*np.pi*1j/Nx )
              exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 1] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
              exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 2] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
  exp_diag_cupy=cp.asarray(exp_diag)
  return exp_diag_cupy

# calculate the 3pt exp part
def phase_exp_3pt(Mom,Nx):
  phase_exp = np.zeros(Nx**3, dtype=complex)
  for z in range(0,Nx):
    for y in range(0,Nx):
      for x in range(0,Nx):
        ni = np.array([z,y,x])
        phase_exp[z*(Nx**2)+y*Nx+x] = np.exp(-2*np.pi*1j*np.dot(Mom,ni)/Nx)
  return phase_exp

# calculate the VdV link part
def VdV_sink_t_link(gauge_link, eig_dir, t, Nev1, conf_id, Nx, link_dir, link_max):
    eigvecs_cupy = readin_eigvecs_device(eig_dir, t, Nev1, conf_id, Nx).reshape(Nev1,Nx,Nx,Nx,3) # (NEV, Z, Y, X, C)
    if (link_dir=='0'):
        VDV_link_cupy = cp.zeros((Nev1,Nev1), dtype=cp.complex128)
        VDV_link_cupy = contract("ab,ca->bc", cp.conj((eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3)).T), eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3))
        #cp.conj(cp.matmul((eigvecs_cupy * phase_exp).reshape(Nev1,Nx*Nx*Nx*3), cp.conj((eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3)).T))).T
    else:
        VDV_link_cupy = cp.zeros((2*link_max+1,Nev1,Nev1), dtype=cp.complex128)
        gauge_link = gauge_link.reshape(Nx,Nx,Nx,3,3)
        if link_dir=='Z':axis_dir=1
        if link_dir=='Y':axis_dir=2
        if link_dir=='X':axis_dir=3
        for link_indx in range(-link_max,link_max+1,1):
            link_rolled = cp.zeros((Nev1,Nx*Nx*Nx,3), dtype=cp.complex128)
            gauge_link_rolled = cp.zeros((Nx*Nx*Nx,3,3), dtype=cp.complex128)
            gauge_link_rolled[:] = cp.identity(3,dtype=cp.complex128)
            eigvecs_link_rolled = cp.roll(eigvecs_cupy,-1*link_indx,axis=axis_dir)
            if link_indx == 0:
                link_rolled = eigvecs_link_rolled.reshape(Nev1,Nx*Nx*Nx,3)
            else:
                if link_indx < 0:
                    for link_indx_2 in range(abs(link_indx)):
                        gauge_link_rolled = gauge_link_rolled @ cp.roll(gauge_link, abs(link_indx)-link_indx_2, axis=(axis_dir-1)).reshape(Nx*Nx*Nx,3,3)
                    gauge_link_rolled = cp.exp(-1.0*cp.log(gauge_link_rolled)) # gauge link inverse integral (-link_indx -> 0 )
                if link_indx > 0:
                    for link_indx_2 in range(abs(link_indx)):
                        gauge_link_rolled = gauge_link_rolled @ cp.roll(gauge_link, -1* link_indx_2, axis=(axis_dir-1)).reshape(Nx*Nx*Nx,3,3)# gauge link forward integral ( 0 -> link_indx )
                for i in range(3):
                    link_rolled[:,:,i] = cp.sum(gauge_link_rolled[:,i,:] * eigvecs_link_rolled.reshape(Nev1,Nx*Nx*Nx,3), axis=-1)
                    # matrix multiplication. Do this just because if use the @ python can't recognition matrix. it's say the matrix dimensions are inconsistent
            VDV_link_cupy[link_indx+link_max] = contract("ab,ca->bc", cp.conj((eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3)).T), (link_rolled.reshape(Nev1,Nx*Nx*Nx*3)))
    return VDV_link_cupy

def Mom_VdV_sink_t(eig_dir, t, Nev1, conf_id, Nx, phase_exp, link_dir,link_max):
    eigvecs_cupy = readin_eigvecs_device(eig_dir, t, Nev1, conf_id, Nx).reshape(Nev1,Nx,Nx,Nx,3) # (NEV, Z, Y, X, C)
    phase_exp = phase_exp.reshape(Nx,Nx,Nx,3)
    if (link_dir=='0'):
      VDV_link_cupy = cp.zeros((Nev1,Nev1), dtype=complex)
      VDV_link_cupy = contract("ab,ca->bc", cp.conj(((eigvecs_cupy * phase_exp).reshape(Nev1,Nx*Nx*Nx*3)).T), eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3))
      # cp.conj(cp.matmul((eigvecs_cupy * phase_exp).reshape(Nev1,Nx*Nx*Nx*3), cp.conj((eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx*3)).T))).T
    else:
      VDV_link_cupy = cp.zeros((Nx,Nev1,Nev1), dtype=complex)
      for dir in range(1,Nx+1):
        if (link_dir=='z'):
          Z = dir ;Y = Nx + 1 ;X = Nx + 1
        if (link_dir=='y'):
          Z = Nx + 1 ;Y = dir ;X = Nx + 1
        if(link_dir=='z'):
          Z = Nx + 1 ;Y = Nx + 1 ;X = dir
        Z0 = (Z-1)%Nx ; Y0 = (Y-1)%Nx ; X0 = (X-1)%Nx
        VDV_link_cupy[dir-1] = contract("ab,ca->bc",cp.conj(((eigvecs_cupy[:,Z0:Z,Y0:Y,X0:X,:]* phase_exp[Z0:Z,Y0:Y,X0:X,:]).reshape(Nev1,Nx*Nx*3)).T),(eigvecs_cupy[:,Z0:Z,Y0:Y,X0:X,:] ).reshape(Nev1,Nx*Nx*3))
        # cp.conj(cp.matmul((eigvecs_cupy[:,Z0:Z,Y0:Y,X0:X,:] * phase_exp[Z0:Z,Y0:Y,X0:X,:]).reshape(Nev1,Nx*Nx*3), cp.conj((eigvecs_cupy[:,Z0:Z,Y0:Y,X0:X,:].reshape(Nev1,Nx*Nx*3)).T))).T
              
    # VDV_cupy = cp.asnumpy(VDV_cupy)
    return VDV_link_cupy

# calculate the VVV mom part
def Mom_VVV_sink_t(eig_dir, t, Nev1, conf_id, Nx, phase_exp, link_dir):
  mid_VVV = readin_eigvecs_device(eig_dir, t, Nev1, conf_id, Nx)
  VVV_timeslice_t = cp.zeros((Nx,Nev1,Nev1,Nev1), dtype=cp.complex128)
  mid_VVV = mid_VVV.reshape(Nev1,Nx,Nx,Nx,3)
  phase_exp = phase_exp.reshape(Nx,Nx,Nx)
  for dir in range(1,Nx+1):
    if (link_dir=='0'):
      Z = dir ;Y = Nx + 1 ;X = Nx + 1
    if (link_dir=='z'):
      Z = dir ;Y = Nx + 1 ;X = Nx + 1
    if (link_dir=='y'):
      Z = Nx + 1 ;Y = dir ;X = Nx + 1
    if(link_dir=='z'):
      Z = Nx + 1 ;Y = Nx + 1 ;X = dir
    Z0 = (Z-1)%Nx ; Y0 = (Y-1)%Nx ; X0 = (X-1)%Nx
    VVV_timeslice_t_1 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2])
    VVV_timeslice_t_2 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0])
    VVV_timeslice_t_3 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1])
    VVV_timeslice_t_4 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1])
    VVV_timeslice_t_5 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2])
    VVV_timeslice_t_6 = contract("zyx,azyx,bzyx,czyx->abc", phase_exp[Z0:Z,Y0:Y,X0:X], mid_VVV[:,Z0:Z,Y0:Y,X0:X,2], mid_VVV[:,Z0:Z,Y0:Y,X0:X,1], mid_VVV[:,Z0:Z,Y0:Y,X0:X,0])
    VVV_timeslice_t[dir-1] = VVV_timeslice_t_1+VVV_timeslice_t_2+VVV_timeslice_t_3-VVV_timeslice_t_4-VVV_timeslice_t_5-VVV_timeslice_t_6
    
  if (link_dir=='0'):
    VVV_timeslice_t = cp.sum(VVV_timeslice_t,axis=0)
  # VVV_timeslice_t = cp.asnumpy(VVV_timeslice_t_1)
  return VVV_timeslice_t