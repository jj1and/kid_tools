import numpy as np
import pandas as pd
from . import functions as fn
from lmfit import minimizer, Parameters, report_fit

class Oneshot():
    def __init__(self, time, phase, ref_swp_fname, trg_fname):
        self.time = time
        self.phase = phase
        #self.amp = np.array([])
        self.phase_fit_range = np.array([])
        self.ref_swp_fname = ref_swp_fname
        self.trg_fname = trg_fname
        self.lmfit_init_tau_params = Parameters()
        self.lmfit_tau_result = minimizer.MinimizerResult()

    def set_init_phase_params(self, tau_lower=1e-6, tau_upper=1000*1e-6, Amp_lower=-np.pi, Amp_upper=np.pi, init_len=50, **kwargs):
        options={'fit_range':False,
        'Amp_min_max':[0.05, 100e-6, 400e-6]}
        options.update(kwargs)
        init_phase_bias = np.mean(self.phase[0:init_len])
        bias_sigma = np.std(self.phase[0:init_len])
        std_phase_sum = np.sum(self.phase[0:init_len])
        if(options['fit_range']==True):
            init_phase_Amp, start, stop = np.array(options['Amp_min_max'])
            init_phase_tau = stop - start
            phase_start_index = np.argmin(np.abs(self.time-start))
            phase_stop_index = np.argmin(np.abs(stop-self.time))
        elif(options['fit_range']==False):
            init_phase_Amp = np.max(np.abs(init_phase_bias - self.phase[init_len:int(len(self.time)/2)]))
            phase_start_index = np.argmax(np.abs(init_phase_bias - self.phase[init_len:int(len(self.time)/2)])) + init_len
            after_signal_indices = np.where(np.abs(init_phase_bias -self.phase[phase_start_index:])<=bias_sigma)
            if(len(after_signal_indices[0])==0):
                phase_stop_index = phase_start_index + 1
            else:
                phase_stop_index = np.min(after_signal_indices)*10+ phase_start_index
                if(phase_stop_index>len(self.time)-1):
                    phase_stop_index = phase_start_index + 1
            init_phase_tau = self.time[np.min(np.where(self.phase[phase_start_index:]>=(init_phase_bias-init_phase_Amp/np.e)))+phase_start_index] - self.time[phase_start_index]
            #phase_stop_index = np.argmin(np.abs(self.time[phase_start_index:]-(self.time[phase_start_index]+10*init_phase_tau)))
            # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
            self.lmfit_init_tau_params.add_many(
            ('phase_Amp', init_phase_Amp, True, Amp_lower, Amp_upper, None, None),
            ('phase_tau', init_phase_tau, True, tau_lower, tau_upper, None, None),
            ('phase_bias', init_phase_bias, True, None, None, None, None),
            ('phase_start_t', self.time[phase_start_index], False, None, None, None, None))
        
        phase_sum = np.sum(self.phase[phase_start_index:phase_start_index+5])
        self.phase_fit_range = np.array([phase_start_index, phase_stop_index])
        return std_phase_sum, phase_sum

    def fit_phase_tau(self):
        #fit_range_indices = np.where((self.time>=self.phase_fit_range[0])&(self.time<=self.phase_fit_range[1]))
        fit_range_indices = np.arange(self.phase_fit_range[0], self.phase_fit_range[1])
        fit_time = self.time[fit_range_indices]
        fit_phase = self.phase[fit_range_indices]
        self.lmfit_tau_result = minimizer.minimize(fn.phase_tau_func_residual, self.lmfit_init_tau_params, args=(fit_time, fit_phase), nan_policy='propagate')
        report_fit(self.lmfit_tau_result)

    def output_data(self):
        return self.time, self.phase

class Trgholder():
    def __init__(self, ref_swp_fname, trg_fname, sample_rate):
        self.oneshot_list = []
        self.failed_list = []
        self.analyzed_data = pd.DataFrame()
        self.sample_rate = sample_rate
        self.ref_swp_fname = ref_swp_fname
        self.trg_fname = trg_fname

    def analyze_trg(self, trg_set, **kwargs):
        options = {'t_area_upper':3.0E-4}
        options.update(kwargs)
        analyzed_data_header = ['phase_tau', 'phase_tau_err', 'phase_Amp', 'phase_Amp_err', 'phase_area', 'phase_area_err', 'phase_bias', 'phase_bias_err']
        analyzed_data_list = [analyzed_data_header]
        for one_trg in trg_set:
            tmp_shot = Oneshot(one_trg[:, 0], one_trg[:, 1], self.ref_swp_fname, self.trg_fname)
            tmp_std_phase_sum, tmp_phase_sum = tmp_shot.set_init_phase_params(**kwargs)
            if(np.diff(tmp_shot.phase_fit_range)[0]<=3):
                self.failed_list.append(tmp_shot)
            elif(np.abs(tmp_phase_sum/tmp_std_phase_sum)<1):
                self.failed_list.append(tmp_shot)
            else:
                tmp_shot.fit_phase_tau()
                if(tmp_shot.lmfit_tau_result.ier in (1, 2, 3)):
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
                        tmp_row = [tmp_phase_tau, tmp_phase_tau_err, tmp_phase_Amp, tmp_phase_Amp_err, tmp_area, tmp_area_err, tmp_phase_bias, tmp_phase_bias_err]
                    else:
                        print("Could not estimate error-bars.")
                        tmp_phase_Amp = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_Amp']
                        tmp_phase_Amp_err = np.nan

                        tmp_phase_tau = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_tau']
                        tmp_phase_tau_err = np.nan

                        tmp_phase_bias = tmp_shot.lmfit_tau_result.params.valuesdict()['phase_bias']
                        tmp_phase_bias_err = np.nan

                        tmp_area = tmp_phase_Amp*tmp_phase_tau*(1-np.exp(-options['t_area_upper']/tmp_phase_tau))
                        tmp_area_err = np.nan
                        tmp_row = [tmp_phase_tau, tmp_phase_tau_err, tmp_phase_Amp, tmp_phase_Amp_err, tmp_area, tmp_area_err, tmp_phase_bias, tmp_phase_bias_err]

                    analyzed_data_list.append(tmp_row)

                else:
                    self.failed_list.append(tmp_shot)

        self.analyzed_data = pd.DataFrame(analyzed_data_list[1:], columns=analyzed_data_list[0])
        print('{0} trg succeeded to fit / {1} trg failed to fit : in {2} trggers'.format(len(self.oneshot_list), len(self.failed_list), trg_set.shape[0]))
        
    def select_trg(self, selected_trg_indices):
        selected_trg = self.oneshot_list[selected_trg_indices]
        return selected_trg


    
