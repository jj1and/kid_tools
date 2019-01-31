import numpy as np
#for fit funcs
def t21_func(f, params):
    r_a, i_a, tau, fr, qr, qc, phi_0 = [params['r_a'], params['i_a'], params['tau'], params['fr'], params['qr'], params['qc'], params['phi_0']]
    val = (r_a+i_a*1j)*np.exp(-2j*np.pi*f*tau)*(1-(np.exp(1j*phi_0)*qr/qc)/(1+2j*qr*(f/fr-1)))
    return val

def t21_func_residual(params, f, ydata, eps_data):
    val = t21_func(f, params)
    residual = (ydata - val)/eps_data
    return residual.view(np.float)


def reduced_chi_sq(y, fit_y, sigma, num_pf_params):
    n = len(y)
    re_chi_sq = np.sum((np.abs(y-fit_y)/sigma)**2)/(n-num_pf_params)
    return re_chi_sq

def theta_func(f, params):
    theta_0, qr, fr = [params['theta_0'], params['qr'], params['fr']]
    val = -theta_0 + 2.0*np.arctan(2.0*qr*(1.0-f/fr))
    return val

def theta_func_resudial(params, f, theta):
    val = theta_func(f, params)
    residual = (theta-val)
    return residual

def Nqp_func(T, params):
    kb = 8.617*1E-5
    N_0, delta_0, volume = [params['N0'], params['delta_0'], params['volume']]
    Nqp = volume*2*N_0 * np.sqrt(2*np.pi*kb*T*1E-3*delta_0)*np.exp(-delta_0/(kb*T*1E-3))
    return Nqp

def phase_Nqp_func(Nqp, params):
    dth_dNqp, phase_bias = [params['dth_dNqp'], params['phase_bias']]
    phase_shift = dth_dNqp*Nqp + phase_bias
    return phase_shift

def phase_Nqp_func_resiual(params, Nqp, phase_shift):
    val = phase_Nqp_func(Nqp, params)
    residual = phase_shift - val
    return residual

def nep_func(phase_psd, f, params):
    eta, delta, dth_dNqp, tau_qp, tau_res = [params['eta'], params['delta'], params['dth_dNqp'], params['tau_qp'], params['tau_res']]
    nep_sq = phase_psd*((eta*tau_qp*dth_dNqp/(delta*1.602*1E-19))**(-2))*(1 + (tau_qp*2*np.pi*f)**2)*(1 + (tau_res*2*np.pi*f)**2)
    return np.sqrt(nep_sq)

def phase_tau_func(time, params):
    A, tau, bias, start_t= [params['phase_Amp'], params['phase_tau'], params['phase_bias'], params['phase_start_t']]
    y = -A*np.exp(-(time-start_t)/tau) + bias
    return y

def phase_tau_func_residual(params, time, y):
    val = phase_tau_func(time, params)
    residual = y-val
    return residual




#def r_cov_func(xc, yc, qr, qc, xc_cov, yc_cov, qr_cov, qc_cov):
#    zc2 = xc**2 + yc**2
#    qr_2qc_qr2_zc2 = ((qr/(2*qc-qr))**2)/zc2
#    zc2_qc2_qr4 = zc2/((2*qc-qr)**4)
#    r_cov2 = qr_2qc_qr2_zc2*((xc_cov*xc)**2 + (yc_cov*yc)**2) + zc2_qc2_qr4*((2*qc*qr_cov)**2+(2*qr*qc_cov)**2)
#    r_cov = np.sqrt(r_cov2)
#    return r_cov