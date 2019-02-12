import numpy as np
import pandas as pd
import scipy.stats as st
from . import functions as fn
from lmfit import minimizer, Parameters, report_fit

class Oneshot():
    def __init__(self, time, phase, ref_swp_fname, trg_fname):
        self.time = time
        self.phase = phase
        #self.amp = np.array([])
        self.signal_flag = False
        self.fh_skew = 0.0
        self.std_ratio = 1.0
        self.phase_fit_range = np.array([])
        self.ref_swp_fname = ref_swp_fname
        self.trg_fname = trg_fname
        self.lmfit_init_tau_params = Parameters()
        self.lmfit_tau_result = minimizer.MinimizerResult()

    def sn_discri(self, **kwargs):
        options={'fit_range':False, 
        'skew_ther':-0.60, 
        'std_ratio_ther':0.80, 
        'divide_index':200,
        'Amp_min_max':[0.05, 100e-6, 400e-6]}
        options.update(kwargs)
        fh_skew = st.skew(self.phase[0:options['divide_index']])
        fh_std = np.std(self.phase[0:options['divide_index']])
        lh_std = np.std(self.phase[options['divide_index']:])
        std_ratio = lh_std/fh_std
        if((fh_skew>options['skew_ther'])&(std_ratio>options['std_ratio_ther'])):
            self.signal_flag = False
        else:
            self.signal_flag = True
        self.fh_skew = fh_skew
        self.std_ratio = std_ratio
        return self.signal_flag

    def set_init_phase_params(self, init_phase_tau = 50e-6, tau_lower=1e-6, tau_upper=1000*1e-6, Amp_lower=-np.pi, Amp_upper=np.pi, header_length=100, **kwargs):
        options={'fit_range':False,
        'Amp_min_max':[0.05, 100e-6, 400e-6]}
        options.update(kwargs)
        init_phase_bias = np.mean(self.phase[0:header_length])
        if(options['fit_range']==True):
            init_phase_Amp, start, stop = np.array(options['Amp_min_max'])
            init_phase_tau = stop - start
            phase_start_index = np.argmin(np.abs(self.time-start))
            phase_stop_index = np.argmin(np.abs(stop-self.time))
        elif(options['fit_range']==False):
            init_phase_Amp = self.phase[header_length]
            phase_start_index = header_length+1
            phase_stop_index = np.argmin(np.abs(self.time[phase_start_index]+init_phase_tau*8-self.time))
            # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
            self.lmfit_init_tau_params.add_many(
            ('phase_Amp', init_phase_Amp, True, Amp_lower, Amp_upper, None, None),
            ('phase_tau', init_phase_tau, True, tau_lower, tau_upper, None, None),
            ('phase_bias', init_phase_bias, True, None, None, None, None),
            ('phase_start_t', self.time[phase_start_index], False, None, None, None, None))
        self.phase_fit_range = np.array([phase_start_index, phase_stop_index])

    def fit_phase_tau(self):
        #fit_range_indices = np.where((self.time>=self.phase_fit_range[0])&(self.time<=self.phase_fit_range[1]))
        fit_range_indices = np.arange(self.phase_fit_range[0], self.phase_fit_range[1])
        fit_time = self.time[fit_range_indices]
        fit_phase = self.phase[fit_range_indices]
        self.lmfit_tau_result = minimizer.minimize(fn.phase_tau_func_residual, self.lmfit_init_tau_params, args=(fit_time, fit_phase), nan_policy='propagate', method='nelder')
        report_fit(self.lmfit_tau_result)

    def output_data(self):
        return self.time, self.phase

class Trgholder():
    def __init__(self, ref_swp_fname, trg_fname, sample_rate):
        self.oneshot_list = []
        self.failed_list = []
        self.analyzed_data = pd.DataFrame()
        self.analyzed_failed_data = pd.DataFrame()
        self.sample_rate = sample_rate
        self.ref_swp_fname = ref_swp_fname
        self.trg_fname = trg_fname

    def analyze_trg(self, trg_set, *args, **kwargs):
        options = {
            't_area_upper':3.0E-4
        }
        options.update(kwargs)
        analyzed_data_header = ['phase_tau', 'phase_tau_err', 'phase_Amp', 'phase_Amp_err', 'phase_area', 'phase_area_err', 'phase_bias', 'phase_bias_err', 'fh_skew', 'std_ratio']
        analyzed_failed_data_header = ['fh_skew', 'std_ratio']
        analyzed_data_list = [analyzed_data_header]
        analyzed_failed_data_list = [analyzed_failed_data_header]
        for one_trg in trg_set:
            tmp_shot = Oneshot(one_trg[:, 0], one_trg[:, 1], self.ref_swp_fname, self.trg_fname)
            tmp_shot.sn_discri(**kwargs)
            tmp_shot.set_init_phase_params(*args, **kwargs)
            if(tmp_shot.signal_flag==False):
                self.failed_list.append(tmp_shot)
                analyzed_failed_data_list.append([tmp_shot.fh_skew, tmp_shot.std_ratio])
            else:
                tmp_shot.fit_phase_tau()
                if(tmp_shot.lmfit_tau_result.redchi<10):
                    print(tmp_shot.lmfit_tau_result.message)
                    if(tmp_shot.lmfit_tau_result.params.valuesdict()['phase_tau']>500e-6):
                        self.failed_list.append(tmp_shot)
                    elif((tmp_shot.lmfit_tau_result.errorbars)==True):
                        self.oneshot_list.append(tmp_shot)
                        tmp_sigs = np.sqrt(np.diag(tmp_shot.lmfit_tau_result.covar))

                        tmp_phase_Amp = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_Amp']
                        tmp_phase_Amp_err = tmp_sigs[0]

                        tmp_phase_tau = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_tau']
                        tmp_phase_tau_err = tmp_sigs[1]

                        tmp_phase_bias = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_bias']
                        tmp_phase_bias_err = tmp_sigs[2]

                        da_dAmp = tmp_phase_tau*(1-np.exp(-options['t_area_upper']/tmp_phase_tau))
                        da_dtau = tmp_phase_Amp*(1-(1+options['t_area_upper']/tmp_phase_tau)*np.exp(-options['t_area_upper']/tmp_phase_tau))
                        tmp_area = np.abs(tmp_phase_Amp*tmp_phase_tau*(1-np.exp(-options['t_area_upper']/tmp_phase_tau)))
                        tmp_area_err = np.sqrt((da_dAmp*tmp_phase_Amp_err)**2+(da_dtau*tmp_phase_tau_err)**2)
                        tmp_row = [tmp_phase_tau, tmp_phase_tau_err, tmp_phase_Amp, tmp_phase_Amp_err, tmp_area, tmp_area_err, tmp_phase_bias, tmp_phase_bias_err, tmp_shot.fh_skew, tmp_shot.std_ratio]
                    else:
                        print("Could not estimate error-bars.")
                        self.oneshot_list.append(tmp_shot)
                        tmp_phase_Amp = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_Amp']
                        tmp_phase_Amp_err = np.nan

                        tmp_phase_tau = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_tau']
                        tmp_phase_tau_err = np.nan

                        tmp_phase_bias = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_bias']
                        tmp_phase_bias_err = np.nan

                        tmp_area = tmp_phase_Amp*tmp_phase_tau*(1-np.exp(-options['t_area_upper']/tmp_phase_tau))
                        tmp_area_err = np.nan
                        tmp_row = [tmp_phase_tau, tmp_phase_tau_err, tmp_phase_Amp, tmp_phase_Amp_err, tmp_area, tmp_area_err, tmp_phase_bias, tmp_phase_bias_err, tmp_shot.fh_skew, tmp_shot.std_ratio]

                    analyzed_data_list.append(tmp_row)

                else:
                    self.failed_list.append(tmp_shot)
                    analyzed_failed_data_list.append([tmp_shot.fh_skew, tmp_shot.std_ratio])

        self.analyzed_data = pd.DataFrame(analyzed_data_list[1:], columns=analyzed_data_list[0])
        self.analyzed_failed_data = pd.DataFrame(analyzed_failed_data_list[1:], columns=analyzed_failed_data_list[0])
        print('{0} trg succeeded to fit / {1} trg failed to fit : in {2} trggers'.format(len(self.oneshot_list), len(self.failed_list), trg_set.shape[0]))
        
    def select_trg(self, selected_trg_indices):
        selected_trg = self.oneshot_list[selected_trg_indices]
        return selected_trg


    
