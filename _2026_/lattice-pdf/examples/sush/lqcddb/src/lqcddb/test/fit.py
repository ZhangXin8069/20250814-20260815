import lsqfit
import numpy as np
import gvar as gv
Nconf = 50
Nt = 72

data_sample = np.random.random((Nconf, Nt)) # Nconf, Nt for jacknife

data_2pt_mean = data_sample.mean(0)
data_2pt_err = data_sample.std(0) * np.sqrt(Nconf - 1)
data_2pt_cov = np.cov(data_sample, rowvar = False, ddof = 1) * (Nconf - 1)

# fit band
X = np.arange(10, 30, 1)

# parameter init
ini_prr = {"Z": "4(1000)", "E": "0.1(1000)"}

def Fit_model(t_fit, p):
    modle = {}
    for name in t_fit.keys():
        ts = t_fit[f'{name}']
        modle[f'{name}'] = ((p["Z"])**2 / (2 * p["E"])) * (np.exp(-1 * p["E"] * ts) + np.exp(-1 *  p["E"] * (Nt - ts)))
    return modle


# just for the mean fit
t_fit = {"Ratio": X}
data_fit = {"Ratio": gv.gvar(data_2pt_mean[X], data_2pt_err[X])}

fit = lsqfit.nonlinear_fit(
    data=(t_fit, data_fit), fcn=Fit_model, prior=ini_prr, debug=True
)

# for the all configuration
t_fit = {"Ratio": X}
E_array = np.zeros((Nconf))
Z_array = np.zeros((Nconf))

for i in range(Nconf):
    data_fit = {"Ratio": gv.gvar(data_2pt_mean[i, 10:30], data_2pt_cov[i, 10:30, 10:30])}
    fit = lsqfit.nonlinear_fit(
        data=(t_fit, data_fit), fcn=Fit_model, prior=ini_prr, debug=True
    )

    # load all parameter
    E_array[i] = fit.p['E'].mean
    Z_array[i] = fit.p['Z'].mean


E_mean = E_array.mean(0) 
E_err  = E_array.std(0) * np.sqrt(Nconf - 1)

Z_mean = Z_array.mean(0)    
Z_err  = Z_array.std(0) * np.sqrt(Nconf - 1)
   
#...