import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
d = cp.cuda.Device(); mem = d.mem_info
props = cp.cuda.runtime.getDeviceProperties(d.id)
name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
print(f'GPU: {name}')
print(f'Mem: {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free')
print(f'CuPy: {cp.__version__}  |  CUDA: {cp.cuda.runtime.runtimeGetVersion()}')
print()

for mod in ['numpy','scipy','matplotlib','cupy','h5py']:
    m = __import__(mod)
    ver = getattr(m, '__version__', '?')
    print(f'  {"✓"} {mod}: {ver}')

print()
paths = [
    ('eigvec', '/public/group/lqcd/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.eigenvector.npy'),
    ('gauge_6250', '/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6250.lime'),
    ('gauge_6450', '/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6450.lime'),
    ('gauge_6650', '/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6650.lime'),
    ('peram_6250', '/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/6250/'),
    ('peram_6450', '/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/6450/'),
    ('peram_6650', '/public/group/lqcd/sunpeng/mom_smear_perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/mz2_my0_mx0/6650/'),
]
all_ok = True
for name, p in paths:
    ok = os.path.exists(p)
    if ok:
        if os.path.isfile(p):
            sz = os.path.getsize(p) / 1024**3
            print(f'  ✓ {name}: ...{p[-60:]} ({sz:.1f} GB)')
        else:
            n = len(os.listdir(p))
            print(f'  ✓ {name}: ...{p[-60:]} ({n} files)')
    else:
        print(f'  ✗ {name}: MISSING')
        all_ok = False

if all_ok:
    print('\nAll data paths accessible — ready to run!')
else:
    print('\nSome data paths missing. Run: bash /root/lattice-pdf/agent/docker/download_beta6.20_mu-0.2770_ms-0.2400_L24x72.sh')
