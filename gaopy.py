import sys
import csv
import numpy as np
from . import functions as fn
from . import utilfunctions as util
from copy import deepcopy 
from scipy import fftpack
from scipy import signal
from lmfit import minimizer, Parameters, report_fit

class Gao():
    def __init__(self, sweep_file_name="sweep.dat",sg_freq=4000):
        self.swp_file_name = sweep_file_name
        self.save_dir = './'
        self.sg = sg_freq
        self.tod_freq = 0.0
        self.fine_fit_redchi_ther = 1.0
        self.comb_freq_amp=np.ndarray([])
        self.comb_freq_phase=np.ndarray([])
        self.theta_fit_range = np.array([])
        self.fine_fit_range = np.array([])
        self.lmfit_init_circle_params = Parameters()
        self.lmfit_circle_params = Parameters()
        self.lmfit_init_theta_params = Parameters()
        self.lmfit_theta_result = minimizer.MinimizerResult()
        self.lmfit_init_params = Parameters()
        self.lmfit_result = minimizer.MinimizerResult()

        #file loading via numpy.genfromtxt
        swp_data = np.genfromtxt(self.swp_file_name, delimiter=" ")
        all_I = swp_data[:,1]
        all_Q = swp_data[:,2]
        #all_amp = swp_data[:,3]
        all_f = self.sg*1.0E6 + swp_data[:,0]
        all_phase = swp_data[:,4]
        self.f = all_f
        self.I = all_I
        self.Q = all_Q
        self.phase = all_phase

    def set_save_dir(self, save_dir='./'):
        self.save_dir = save_dir

    def get_fit_params(self, *fit_params, **kwargs):
        options={"load_fit_file":"none"}
        options.update(kwargs)
        circle_params_dict = self.lmfit_circle_params.valuesdict()
        try:
            if(options["load_fit_file"]!="none"):
                tau, xc, yc, r, fr, Qr, Qc, phi_0 = util.fit_file_reader(options["load_fit_file"])
            elif(fit_params!=()):
                tau, xc, yc, r, fr, Qr, phi_0 = fit_params 
            elif(self.lmfit_result.redchi<self.fine_fit_redchi_ther ):
                params_dict = self.lmfit_result.params.valuesdict()
                tau = params_dict['tau']
                fr = params_dict['fr']
                Qr = params_dict['qr']
                phi_0 = params_dict['phi_0']
                xc = circle_params_dict['xc']
                yc = circle_params_dict['yc']
                r = circle_params_dict['r']
            else:
                print("fine fitting seems to be failed")
                params_dict = self.lmfit_init_params.valuesdict()
                circle_params_dict = self.lmfit_init_circle_params.valuesdict()
                tau = params_dict['tau']
                fr = params_dict['fr']
                Qr = params_dict['qr']
                phi_0 = params_dict['phi_0']
                xc = circle_params_dict['xc']
                yc = circle_params_dict['yc']
                r = circle_params_dict['r']

        except AttributeError:
            print("Error! Cannot get fine fit params")
            params_dict = self.lmfit_init_params.valuesdict()
            circle_params_dict = self.lmfit_init_circle_params.valuesdict()
            tau = params_dict['tau']
            fr = params_dict['fr']
            Qr = params_dict['qr']
            phi_0 = params_dict['phi_0']
            xc = circle_params_dict['xc']
            yc = circle_params_dict['yc']
            r = circle_params_dict['r']

        return tau, xc, yc, r, fr, Qr, phi_0

    def save_fine_fit_params(self, save_fname1='fine_fit_params.csv', save_fname2='fine_fit_circle_params.csv'):
        try:
            with open(self.save_dir+save_fname1, 'w', newline="") as f:
                csv_header = ['para_name', 'value', 'sigma', 'init_val']
                csv_rows = [csv_header]
                var_names = self.lmfit_result.var_names
                val_dict = self.lmfit_result.params.valuesdict()
                sig = np.sqrt(np.diag(self.lmfit_result.covar))
                init_vals =  self.lmfit_result.init_vals
                cnt = 0
                for key in val_dict.keys():
                    if(key=='qi'):
                        dqi_dqc_2 = (val_dict['qr']/(val_dict['qc']-val_dict['qr']))**4
                        dqi_dqr_2 = (val_dict['qc']/(val_dict['qc']-val_dict['qr']))**4
                        sig_qc = sig[var_names.index('qc')]
                        sig_qr = sig[var_names.index('qr')]
                        sig_qi = np.sqrt(dqi_dqc_2*sig_qc**2 + dqi_dqr_2*sig_qr**2)
                        tmp_row = [key, str(val_dict[key]), str(sig_qi), 'None']
                    else:
                        tmp_row = [key, str(val_dict[key]), str(sig[cnt]), str(init_vals[cnt])]
                        cnt += 1
                    csv_rows.append(tmp_row)
                writer = csv.writer(f)
                for row in csv_rows:
                    writer.writerow(row)

            with open(self.save_dir+save_fname2, 'w', newline="") as f  :
                csv_header = ['para_name', 'value']
                csv_rows = [csv_header]
                val_dict = self.lmfit_circle_params.valuesdict()
                for key in val_dict.keys():
                    tmp_row = [key, str(val_dict[key])]
                    csv_rows.append(tmp_row)
                writer = csv.writer(f)
                for row in csv_rows:
                    writer.writerow(row)
        except AttributeError:
            print("Error! Cannot get fine fit params")


    def save_coarse_fit_params(self, save_fname1='coarse_fit_params.csv', save_fname2='coarse_fit_circle_params.csv'):
        with open(self.save_dir+save_fname1, 'w', newline="") as f  :
            csv_header = ['para_name', 'value']
            csv_rows = [csv_header]
            val_dict = self.lmfit_init_params.valuesdict()
            for key in val_dict.keys():
                tmp_row = [key, str(val_dict[key])]
                csv_rows.append(tmp_row)
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)
        with open(self.save_dir+save_fname2, 'w', newline="") as f  :
            csv_header = ['para_name', 'value']
            csv_rows = [csv_header]
            val_dict = self.lmfit_init_circle_params.valuesdict()
            for key in val_dict.keys():
                tmp_row = [key, str(val_dict[key])]
                csv_rows.append(tmp_row)
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)

    def output_fine_params(self):
        params_header = ['para_name', 'value', 'sigma', 'init_val']
        params_rows = [params_header]
        circ_params_header = ['para_name', 'value']
        circ_params_rows = [circ_params_header]
        try:
            var_names = self.lmfit_result.var_names
            val_dict = self.lmfit_result.params.valuesdict()
            sig = np.sqrt(np.diag(self.lmfit_result.covar))
            init_vals =  self.lmfit_result.init_vals
            cnt = 0
            for key in val_dict.keys():
                if(key=='qi'):
                    dqi_dqc_2 = (val_dict['qr']/(val_dict['qc']-val_dict['qr']))**4
                    dqi_dqr_2 = (val_dict['qc']/(val_dict['qc']-val_dict['qr']))**4
                    sig_qc = sig[var_names.index('qc')]
                    sig_qr = sig[var_names.index('qr')]
                    sig_qi = np.sqrt(dqi_dqc_2*sig_qc**2 + dqi_dqr_2*sig_qr**2)
                    tmp_row = [key, str(val_dict[key]), str(sig_qi), 'None']
                else:
                    tmp_row = [key, str(val_dict[key]), str(sig[cnt]), str(init_vals[cnt])]
                    cnt += 1
                params_rows.append(tmp_row)
            
            val_dict = self.lmfit_circle_params.valuesdict()
            for key in val_dict.keys():
                tmp_row = [key, str(val_dict[key])]
                circ_params_rows.append(tmp_row)

        except AttributeError:
            print("cannot get fine params")
        return params_rows, circ_params_rows
    

    def output_coarse_params(self):
        params_header = ['para_name', 'value']
        params_rows = [params_header]
        val_dict = self.lmfit_init_params.valuesdict()
        for key in val_dict.keys():
            tmp_row = [key, str(val_dict[key])]
            params_rows.append(tmp_row)
        circ_params_header = ['para_name', 'value']
        circ_params_rows = [circ_params_header]
        val_dict = self.lmfit_init_circle_params.valuesdict()
        for key in val_dict.keys():
            tmp_row = [key, str(val_dict[key])]
            circ_params_rows.append(tmp_row)
        return params_rows, circ_params_rows


    def phase_smoother(self, theta, **kwargs):
        options = {'std_theta':theta[0]}
        options.update(kwargs)
        mod_theta = deepcopy(theta - options['std_theta'])
        diff_theta =  np.diff(theta)
        ther = 0.5
        #smp_len = 10
        jumped_indices = np.array([],dtype=int)
        jump_cnt = 0
        #diff_std = np.std(diff_theta[0:smp_len])
        for i in range(len(diff_theta)):
            if(np.abs(diff_theta[i]/(2*np.pi))>ther):
                jumped_indices = np.append(jumped_indices, [i], axis=0)
            else:
                pass
        #print("theta_jump found at ", jumped_indices)
        for jumped_index in jumped_indices:
            if(diff_theta[jumped_index]>0):
                mod_theta[jumped_index+1:] -= 2*np.pi
            elif(diff_theta[jumped_index]<0):
                mod_theta[jumped_index+1:] += 2*np.pi
            jump_cnt += 1
        #print("theta_jump: " + str(jump_cnt)+ " times found")

        # diff_max_index = np.argmax(diff_theta)
        # smp_l = 1
        # diff_std = np.std(np.array([diff_theta[diff_max_index-smp_l], diff_theta[diff_max_index+smp_l]]))
        # if(np.abs(diff_theta[diff_max_index])>5*diff_std):
        #     diff_max = diff_theta[diff_max_index]
        #     thre= diff_max*0.8
        #     jumped_indices = np.where(diff_theta>thre)
        #     jump_cnt = 0
        #     for jumped_index in jumped_indices[0]:
        #         if(diff_theta[jumped_index]>0):
        #             mod_theta[jumped_index+1:] -= 2*np.pi
        #         elif(diff_theta[jumped_index]<0):
        #             mod_theta[jumped_index+1:] += 2*np.pi
        #         jump_cnt += 1
        #     print("theta_jump: " + str(jump_cnt)+ " times found") 
        #else:
        #    pass
        return mod_theta

    def remove_tau_effect(self, I, Q, f, tau):
        #calc cable delay effect by using tau value
        I_tau_effect = np.cos(2.0*np.pi*f*tau)
        Q_tau_effect = np.sin(2.0*np.pi*f*tau)

        #calc value for fitting; x,y
        x = I*I_tau_effect - Q*Q_tau_effect
        y = I*Q_tau_effect + Q*I_tau_effect
        return x, y

    def calc_xc_yc(self, x, y):
        w = x**2 + y**2
        M = np.matrix(np.ones((4,4)))
        M[0,0] = np.sum(w**2)
        M[0,1] = np.sum(x*w)
        M[0,2] = np.sum(y*w)
        M[0,3] = np.sum(w)
        M[1,1] = np.sum(x**2)
        M[1,2] = np.sum(x*y)
        M[1,3] = np.sum(x)
        M[2,2] = np.sum(y**2)
        M[2,3] = np.sum(y)
        M[3,3] = w.shape[0]

        M[1,0] = M[0,1]
        M[2,0] = M[0,2]
        M[3,0] = M[0,3]
        M[2,1] = M[1,2]
        M[3,1] = M[1,3]
        M[3,2] = M[2,3]
        
        #calc eigen vector (which is mat_A)
        mat_B = np.matrix([[0,0,0,-2],[0,1,0,0],[0,0,1,0],[-2,0,0,0]])
        inv_B = np.linalg.inv(mat_B)

        eta_array, A_array = np.linalg.eig(inv_B*M)
        min_eta = np.max(eta_array)
        for eta in eta_array:
            if(eta>0):
                if(min_eta>eta):
                    min_eta = eta
                else:
                    pass
            else:
                pass

        min_eta_index = np.where(eta_array==min_eta)

        A = float(A_array[0, min_eta_index])
        B = float(A_array[1, min_eta_index])
        C = float(A_array[2, min_eta_index])
        D = float(A_array[3, min_eta_index])
        R = np.sqrt(B**2 + C**2 - 4*A*D)

        xc = -B/(2*A)
        yc = -C/(2*A)
        r = R/np.abs(2.0*A)
        return xc, yc, r


    def set_data_default_position(self, I, Q, f, *fit_params, **kwargs):
        options = {"coarse_fit":False,
                    "xc":0.0,
                    "yc":0.0}
        options.update(kwargs)
        if(options["coarse_fit"]==True):
            xc = options["xc"]
            yc = options["yc"]
            x = I
            y = Q
        else:
            tau, xc, yc, r, fr, Qr, phi_0 = self.get_fit_params(*fit_params, **kwargs)
            x, y = self.remove_tau_effect(I, Q, f, tau)

        alpha = np.arctan2(yc, xc)
        c_c = np.cos(-alpha)
        s_c = np.sin(-alpha)
        xc_c = (xc - x)*c_c - (yc - y)*s_c
        yc_c = (xc - x)*s_c + (yc - y)*c_c

        return xc_c, yc_c

    def set_fit_range(self, swp_data, fr_MHz, min_MHz, max_MHz):
        fr_Hz = fr_MHz*1.0E6
        min_Hz = min_MHz*1.0E6
        max_Hz = max_MHz*1.0E6
        temp_f = self.sg*1.0E6 + swp_data[:, 0]
        min_freq = fr_Hz - min_Hz
        max_freq = fr_Hz + max_Hz

        fit_range_index = np.where((min_freq<=temp_f)&(temp_f<=max_freq))
        cut_swp_data = swp_data[fit_range_index]
        return cut_swp_data

    def set_fine_fit_range(self, I, Q, f, fr_MHz, min_MHz, max_MHz):
        fr_Hz = fr_MHz*1.0E6
        min_Hz = min_MHz*1.0E6
        max_Hz = max_MHz*1.0E6
        temp_f = f
        min_freq = fr_Hz - min_Hz
        max_freq = fr_Hz + max_Hz

        fit_range_index = np.where((min_freq<=temp_f)&(temp_f<=max_freq))
        fit_I = I[fit_range_index]
        fit_Q = Q[fit_range_index]
        fit_f = f[fit_range_index]
        return fit_I, fit_Q, fit_f

    def coarse_fit(self, **kwargs):
        print("excecuting coarse fit...")
        print("------------------------------------------")
        options={"save_csv":False,
                 "cs_fit_range":False,
                 "cs_fr_min_max":[4010.0, 1.5, 1.5]}
        options.update(kwargs)

        if(options["cs_fit_range"]==True):
            fr_MHz, min_MHz, max_MHz = options["cs_fr_min_max"]
            fr_Hz = fr_MHz*1.0E6
            min_Hz = min_MHz*1.0E6
            max_Hz = max_MHz*1.0E6
            min_freq = fr_Hz - min_Hz
            max_freq = fr_Hz + max_Hz
            fit_range_index = np.where((min_freq<=self.f)&(self.f<=max_freq))
            f = self.f[fit_range_index]
            I = self.I[fit_range_index]
            Q = self.Q[fit_range_index]
        else:
            f = self.f
            I = self.I
            Q = self.Q

        phase = self.phase

        #calc tau delay
        n1 = int(f.shape[0]/20)
        df = f[n1]-f[0]
        if(phase[n1]>phase[0]):
            dp = -2*np.pi+phase[n1]-phase[0]
        else:
            dp = phase[n1]-phase[0]
        tau = -dp/(2*np.pi*df)

        #calc value for fitting; x,y
        x, y = self.remove_tau_effect(I, Q, f, tau) 
        xc, yc, r = self.calc_xc_yc(x, y)

        #centering the circle
        alpha = np.arctan2(yc, xc)
        xc_c, yc_c = self.set_data_default_position(x, y, f, coarse_fit=True, xc=xc, yc=yc, r=r, tau=tau)

        theta = np.arctan2(yc_c, xc_c)
        
        #set the phase continuous
        theta_c = self.phase_smoother(theta, **kwargs)

        #get freq giving minimum amp as temporary reso_freq
        # grad_theta_c = np.gradient(theta_c, f)
        # temp_fr_index = np.argmax(np.abs(grad_theta_c))
        temp_fr_index = np.argmin(np.sqrt(I**2 +Q**2))
        fr = f[temp_fr_index]

        # grad2_theta_c = np.gradient(grad_theta_c, f)
        # span = np.argmax(np.abs(grad2_theta_c[temp_fr_index:]))+temp_fr_index - np.argmax(np.abs(grad2_theta_c[:temp_fr_index]))
        # fs_grad2_absmax_index = np.argmax(np.abs(grad2_theta_c[:temp_fr_index]))
        # lr_grad2_absmax_index = np.argmax(np.abs(grad2_theta_c[temp_fr_index:]))+temp_fr_index
        # span =  lr_grad2_absmax_index - fs_grad2_absmax_index
        # fit_lower_limit_index = fs_grad2_absmax_index-int(span*0.5)
        # fit_upper_limit_index = lr_grad2_absmax_index+int(span*0.5)

        #temp_fr_index = np.argmin(swp_data[:, 3])
        #fr = f[temp_fr_index]
        fit_lower_limit_index = temp_fr_index-int(x.shape[0]/5)
        fit_upper_limit_index = temp_fr_index+int(x.shape[0]/5)


        fit_f = f[fit_lower_limit_index:fit_upper_limit_index]
        fit_theta_c = theta_c[fit_lower_limit_index:fit_upper_limit_index]
        #calc theta_0 and Qr for phi_0 and Q_c
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        self.lmfit_init_theta_params.add_many(
        ('theta_0', 0.0, True, None, None, None, None),
        ('qr', 10000.0, True, None, None, None, None),
        ('fr', fr, True, None, None, None, None))

        self.lmfit_theta_result = minimizer.minimize(fn.theta_func_resudial, self.lmfit_init_theta_params, args=(fit_f, fit_theta_c))
        print("theta_func fit report from lmfit")
        report_fit(self.lmfit_theta_result)
        
        theta_0 = self.lmfit_theta_result.params.valuesdict()['theta_0']
        Qr = self.lmfit_theta_result.params.valuesdict()['qr']
        fr = self.lmfit_theta_result.params.valuesdict()['fr']

        phi_0 = theta_0 - alpha
        Qc = Qr*(np.sqrt(xc**2+yc**2)+r)/(2*r)
        Qi = Qr*Qc/(Qc-Qr)

        f1_index = np.argmin(np.abs(fr-fr/Qr/2 -self.f))
        #a = (self.I[0]+self.Q[0]*1j)*np.exp(2j*np.pi*self.f[0]*tau)/(1-Qr*fr*np.exp(1j*phi_0)/(Qc*fr+2j*Qr*Qc*(self.f[0]-fr)))
        a = (self.I[f1_index]+self.Q[f1_index]*1j)*np.exp(2j*np.pi*self.f[f1_index]*tau)/(1-Qr*fr*np.exp(1j*phi_0)/(Qc*fr+2j*Qr*Qc*(self.f[f1_index]-fr)))
        r_a = np.real(a)
        i_a = np.imag(a)

        # if(options["save_csv"]==True):
        #     #save fit params
        #     fit_contents = "fit params data \n"
        #     tau_cont = "tau, "+ str(tau) + "\n"
        #     xc_cont = "xc, " + str(xc) + "\n"
        #     yc_cont = "yc, " + str(yc) + "\n"
        #     r_cont = "r, " + str(r) + "\n"
        #     fr_cont = "fr, " + str(fr) + "\n"
        #     Qr_cont = "Q, " + str(Qr) + "\n"
        #     Qc_cont = "Qc, " + str(Qc) + "\n"
        #     ph0_cont = "phi_0, " + str(phi_0)
        #     for cont in [tau_cont, xc_cont, yc_cont, r_cont, fr_cont, Qr_cont, Qc_cont, ph0_cont]:
        #         fit_contents = fit_contents + cont
        #     util.data_writer("fit_data.dat", fit_contents)
        # else:
        #     pass

        fit_params = np.array([tau, xc, yc, r, fr, Qr, phi_0])

        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        self.lmfit_init_params.add_many(
        ('r_a', float(r_a), True, None, None, None, None), 
        ('i_a', float(i_a), True, None, None, None, None), 
        ('tau', float(tau), True, None, None, None, None), 
        ('fr', float(fr), True, None, None, None, None), 
        ('qr', float(Qr), True, None, None, None, None), 
        ('qc', float(Qc), True, None, None, None, None), 
        ('qi', float(Qi), False, None, None, None, None), 
        ('phi_0', float(phi_0), True, None, None, None, None) )
        self.lmfit_init_circle_params.add_many(
        ('xc', xc, False, None, None, None, None),
        ('yc', yc, False, None, None, None, None),
        ('r', r, False, None, None, None, None)
        )

        print("\ncoarse fit results")
        self.lmfit_init_params.pretty_print(colwidth=1, columns=['value'])
        print("\ncoarse fit circle property")
        self.lmfit_init_circle_params.pretty_print(colwidth=1, columns=['value'])
        self.theta_fit_range = np.array([np.min(fit_f), np.max(fit_f)])
        return fit_params

    def fine_fit(self, **kwargs):
        print("\nexecuting fine fit...")
        print("------------------------------------------")
        options={"save_csv":False,
                 "fn_fit_range":False,
                 "fn_fr_min_max":[self.lmfit_init_params.valuesdict()['fr']/1.0E6, 0.1, 0.2],
                 "red_chi_ther":self.fine_fit_redchi_ther}
        options.update(kwargs)
        if(options["red_chi_ther"]!=self.fine_fit_redchi_ther):
            self.fine_fit_redchi_ther = options["red_chi_ther"]

        if(options["fn_fit_range"]==True):
            fr_MHz, min_MHz, max_MHz = options["fn_fr_min_max"]
            
        else:
            fr_MHz = self.lmfit_init_params.valuesdict()['fr']/1.0e6
            qr = self.lmfit_init_params.valuesdict()['qr']
            min_MHz = (fr_MHz/qr)
            max_MHz = (fr_MHz/qr)
            # min_MHz = (fr_MHz-self.theta_fit_range[0]/1.0e6)/2
            # max_MHz = (self.theta_fit_range[1]/1.0e6-fr_MHz)/2
        fit_I, fit_Q, fit_f = self.set_fine_fit_range(self.I, self.Q, self.f, fr_MHz, min_MHz, max_MHz)

        #calc eps_data range wa nantonaku
        eps_IQ = self.I[0:int(len(self.I)/100)] + 1j*self.Q[0:int(len(self.I)/100)]
        eps = np.std(eps_IQ)

        eps_data = eps*np.ones(len(fit_f))
        ydata = fit_I + fit_Q*1j
        xdata = fit_f

        for cnt in range(5):
            if(cnt==0):
                proto_lmfit_result_params = self.lmfit_init_params
            
            if(cnt<2):
                lmfit_result = minimizer.minimize(fn.t21_func_residual, proto_lmfit_result_params, args=(xdata, ydata, eps_data))
            elif(cnt>=2):
                lmfit_result = minimizer.minimize(fn.t21_func_residual, proto_lmfit_result_params, args=(xdata, ydata, eps_data), method='powell')
            
            if(cnt==4):
                self.lmfit_result = lmfit_result
            
            if(lmfit_result.redchi<self.fine_fit_redchi_ther):
                self.lmfit_result = lmfit_result
                break
            else:
                proto_lmfit_result_params = lmfit_result.params
        #self.lmfit_result = minimizer.minimize(fn.t21_func_residual, proto1_lmfit_result.params, args=(xdata, ydata, eps_data), method='powell')

        #exp_IQ = fn.t21_func(fit_f, self.lmfit_result.params)
        params_dict = self.lmfit_result.params.valuesdict()

        fine_tau, fine_fr, fine_Qr, fine_Qc, fine_phi_0 = np.array([params_dict['tau'], params_dict['fr'], params_dict['qr'], params_dict['qc'], params_dict['phi_0']])
        fine_Qi = fine_Qr*fine_Qc/(fine_Qc-fine_Qr)
        temp_x, temp_y = self.remove_tau_effect(self.I, self.Q, self.f, fine_tau)
        fine_xc, fine_yc, fine_r = self.calc_xc_yc(temp_x, temp_y)
        
        self.lmfit_result.params.add_many(('qi', fine_Qi, False, None, None, None, None))
        self.lmfit_circle_params.add_many(
        ('xc', fine_xc, False, None, None, None, None),
        ('yc', fine_yc, False, None, None, None, None),
        ('r', fine_r, False, None, None, None, None)
        ) 
        print("\nfine fit report from lmfit")
        report_fit(self.lmfit_result)

        print("\nfine fit circle property")
        self.lmfit_circle_params.pretty_print(colwidth=1, columns=['value'])
        self.fine_fit_range = np.array([np.min(fit_f), np.max(fit_f)])
        fit_params = np.array([fine_tau, fine_xc, fine_yc, fine_r, fine_fr, fine_Qr, fine_phi_0])
        return fit_params


    def tod2psd(self, tod_file_name, sample_rate, tod_freq, *fit_params, **kwargs):
        #sample rate unit is kHz
        tod_data = np.genfromtxt(tod_file_name, delimiter=" ")
        num = tod_data[:, 0]

        options={"load_fit_file":"none",
                 "psd_method":"welch",
                 "segment_rate":50,
                 "ps_option":False}
        options.update(kwargs)

        if((options["segment_rate"]>100)|(options["segment_rate"]<0)):
            sys.exit("overlap rate value range is 0 to 100")

        tau, xc, yc, r, fr, Qr, phi_0 = self.get_fit_params(*fit_params, **kwargs)

        # why??
        tod_I = tod_data[:, 1]*sample_rate
        tod_Q = tod_data[:, 2]*sample_rate

        x, y = self.remove_tau_effect(tod_I, tod_Q, tod_freq, tau)
        xc_c, yc_c = self.set_data_default_position(tod_I, tod_Q, tod_freq)
        theta = np.arctan2(yc_c, xc_c)
        
        phase = self.phase_smoother(theta, **kwargs)
        amp = np.sqrt((xc-x)**2 + (yc-y)**2)/r

        dt = 0.001/sample_rate
        N = num.shape[0]

        if(options["ps_option"]==True):
            han_win = np.hanning(N)
            win_amp = amp * han_win
            win_phase = phase * han_win

            fft_freqlist = fftpack.fftfreq(N, dt)
            fft_pos_freq_list = fft_freqlist[np.where(fft_freqlist>0)]

            fft_amp = fftpack.fft(win_amp)[np.where(fft_freqlist>0)]
            fft_phase = fftpack.fft(win_phase)[np.where(fft_freqlist>0)]
        
            #correction for estimate true power spec
            amp_ps = 2*(np.abs(fft_amp)**2)*(8.0/3.0/N)
            phase_ps = 2*(np.abs(fft_phase)**2)*(8.0/3.0/N)

            #psd = ps/df (df=1/(N*dt))
            ps_data = np.hstack((amp_ps.reshape(-1,1),phase_ps.reshape(-1,1)))
            
            freq_ps = np.hstack((fft_pos_freq_list.reshape(-1,1), ps_data))
            return freq_ps
        
        elif(options["ps_option"]==False):
            if(options["psd_method"]=="periodogram"):
                #if(options["nperseg"]!=0):print("nperseg option is ignored at periodogram method")

                amp_freq, amp_psd = signal.periodogram(amp, sample_rate*1000, window='hann')
                phase_freq, phase_psd = signal.periodogram(phase, sample_rate*1000, window='hann')
            
            elif(options["psd_method"]=="welch"):
                n_perseg = int((options["segment_rate"]/100)*num.shape[0])
                amp_freq, amp_psd = signal.welch(amp, sample_rate*1000, nperseg=n_perseg)
                phase_freq, phase_psd = signal.welch(phase, sample_rate*1000, nperseg=n_perseg)

            else:
                sys.exit("Error! invailed psd estimating method option")

            # fs=sampling rate(ksps)*1000 and Nyquist freq is fs/2. so, cutoff freq is sample_rate*500
            freq_amp = np.hstack((amp_freq[np.where(amp_freq<sample_rate*500)].reshape(-1,1), amp_psd[np.where(amp_freq<sample_rate*500)].reshape(-1, 1)))
            freq_phase = np.hstack((phase_freq[np.where(phase_freq<sample_rate*500)].reshape(-1,1), phase_psd[np.where(phase_freq<sample_rate*500)].reshape(-1, 1)))
            return freq_amp, freq_phase
        
        else:
            sys.exit("Error! invailed ps_option")

    def comb_psd(self, tod_freq, tod_file_list=[["tod_1ksps.dat", 1.0],["tod_100ksps.dat", 100.0], ["tod_1Msps.dat", 1000.0]], **kwargs):
        options = {"psd_unit":"liner",
        "save_to_menber":True}
        options.update(kwargs)

        self.tod_freq = tod_freq
        comb_freq_amp, comb_freq_phase = self.tod2psd(tod_file_list[0][0], tod_file_list[0][1], self.tod_freq, **options)
        cnt = 0
        for tod_file in tod_file_list[1:]:
            temp_freq_amp, temp_freq_phase =  self.tod2psd(tod_file[0], tod_file[1], self.tod_freq, **options)
            comb_freq_amp = np.vstack((comb_freq_amp, temp_freq_amp[np.where(temp_freq_amp[:,0]>=tod_file_list[cnt][1]*500)]))
            comb_freq_phase = np.vstack((comb_freq_phase, temp_freq_phase[np.where(temp_freq_phase[:,0]>=tod_file_list[cnt][1]*500)]))
            cnt += 1
        if(options["save_to_menber"]==True):
            self.comb_freq_amp = comb_freq_amp
            self.comb_freq_phase = comb_freq_phase
        else:
            pass
        if(options["psd_unit"]=="dB"):
                dBcomb_amp = 10*np.log10(comb_freq_amp[:, 1])
                dBcomb_phase = 10*np.log10(comb_freq_phase[:, 1])
                dBcomb_freq_amp = np.hstack((comb_freq_amp[:,0].reshape(-1,1), dBcomb_amp.reshape(-1,1)))
                dBcomb_freq_phase = np.hstack((comb_freq_phase[:,0].reshape(-1,1), dBcomb_phase.reshape(-1,1)))
        elif(options["psd_unit"]=="liner"):
            pass
        else:
            sys.exit("Error! invailed psd unit option")
        return dBcomb_freq_amp, dBcomb_freq_phase

    def save_comb_psd(self, amp_psd_fname='amp_psd.dat', phase_psd_fname='phase_psd.dat'):
        np.savetxt(self.save_dir+amp_psd_fname, self.comb_freq_amp, delimiter=' ', header='freq[Hz] amp_PSD[dBc/Hz]')
        np.savetxt(self.save_dir+phase_psd_fname, self.comb_freq_phase, delimiter=' ', header='freq[Hz] phase_PSD[dBc/Hz]')



class Gaonep(Gao):
    def __init__(self, ref_swp_fname, sg_freq=4000):
        self.oncho_fit_range = np.array([])
        self.Tarray = np.ndarray([])
        self.Tstart_stop = np.ndarray([])
        self.Nqp = np.ndarray([])
        self.phase_shift = np.ndarray([])
        self.nep = np.ndarray([])
        self.lmfit_element_props = Parameters()
        self.lmfit_oncho_init_params = Parameters()
        self.lmfit_oncho_result = minimizer.MinimizerResult()
        self.lmfit_nep_props = Parameters()

        super().__init__(ref_swp_fname, sg_freq)

    def oncho_analisys(self, oncho_file_list, fr, qr):
        self.oncho_file_list = oncho_file_list
        
        tau_res = qr/(np.pi*fr)
        self.lmfit_nep_props.add_many(
        ('tau_res', tau_res, False, None, None, None, None))

        cnt = 0
        Tarray = np.array([])
        Tstart_stop = np.array([])
        phase_shift = np.array([])
        for ele in self.oncho_file_list:
            fname = ele[0]
            temp = ele[1]
            swp_data = np.genfromtxt(fname, delimiter=" ")
            tmp_I = swp_data[:,1]
            tmp_Q = swp_data[:,2]
            tmp_f = self.sg*1.0E6 + swp_data[:,0]
            tmp_fr_idx = np.abs(tmp_f-fr).argmin()
            tmp_xc_c, tmp_yc_c = self.set_data_default_position(tmp_I, tmp_Q, tmp_f)
            tmp_theta = np.arctan2(tmp_yc_c, tmp_xc_c)
            if(cnt==0):
                std_theta = tmp_theta[tmp_fr_idx]
                Tstart_stop = np.array([[ele[2], ele[3]]])
            else:
                Tstart_stop = np.append(Tstart_stop, [[ele[2], ele[3]]], axis=0)
            tmp_theta_c = self.phase_smoother(tmp_theta, std_theta=std_theta)
            # be careful to phase_shift value's sign
            tmp_phase_shift = -tmp_theta_c[tmp_fr_idx]
            phase_shift = np.append(phase_shift, tmp_phase_shift)
            Tarray = np.append(Tarray, temp)
            cnt += 1

        sort = np.argsort(Tarray)
        Tarray = Tarray[sort]
        Tstart_stop = Tstart_stop[sort]
        phase_shift = phase_shift[sort] -phase_shift[sort[0]]

        grad = np.diff(phase_shift)
        ther = 0.8
        for i in range(1, phase_shift.shape[0]-1):
            if(np.abs(grad[i])/(2*np.pi)>ther):
                if(grad[i]<0):
                    phase_shift[i+1:] += 2*np.pi
                elif(grad[i]>0):
                    phase_shift[i+1:] -= 2*np.pi
            else:
                pass

        self.Tarray = Tarray
        self.Tstart_stop = Tstart_stop
        self.phase_shift = phase_shift
        return Tarray, Tstart_stop, phase_shift

    def Temp2Nqp(self, N0, delta_0, volume):
        Nqp = np.array([])
        self.lmfit_element_props.add_many(
        ('N0', N0, False, None, None, None, None), 
        ('delta_0', delta_0, False, None, None, None, None), 
        ('volume', volume, False, None, None, None, None))
        Nqp = fn.Nqp_func(self.Tarray, self.lmfit_element_props)
        self.Nqp = Nqp
        return Nqp

    def load_psd(self, amp_psd_fname, phase_psd_fname, skip_header=1):
        self.comb_freq_amp = np.genfromtxt(amp_psd_fname, delimiter=" ", skip_header=skip_header)
        self.comb_freq_phase = np.genfromtxt(phase_psd_fname, delimiter=" ", skip_header=skip_header)

    def get_psd(self, tod_freq, tod_file_list, **kwargs):
        self.comb_psd(tod_freq, tod_file_list=tod_file_list, **kwargs)


    def calc_nep(self, delta, eta, tau_qp, fit_Nqp_min, fit_Nqp_max, init_dth_dNqp=1.0, init_phase_bias=0.0):
        self.oncho_fit_range = np.array([fit_Nqp_min, fit_Nqp_max])
        self.lmfit_oncho_init_params.add_many(
        ('dth_dNqp', init_dth_dNqp, True, None, None, None, None), 
        ('phase_bias', init_phase_bias, True, None, None, None, None))
        
        fit_Nqp_idices = np.where((fit_Nqp_min<=self.Nqp)&(fit_Nqp_max>=self.Nqp))
        fit_Nqp = self.Nqp[fit_Nqp_idices]
        fit_phase_shift = self.phase_shift[fit_Nqp_idices]
        
        self.lmfit_oncho_result = minimizer.minimize(fn.phase_Nqp_func_resiual, self.lmfit_oncho_init_params, args=(fit_Nqp, fit_phase_shift))
        report_fit(self.lmfit_oncho_result)

        self.lmfit_element_props.add_many(
        ('eta', eta, False, None, None, None, None), 
        ('delta', delta, False, None, None, None, None), 
        ('tau_qp', tau_qp, False, None, None, None, None))

        self.lmfit_nep_props.add_many(
        ('eta', eta, False, None, None, None, None),
        ('delta', delta, False, None, None, None, None),
        ('tau_qp', tau_qp, False, None, None, None, None),
        ('dth_dNqp', self.lmfit_oncho_result.params['dth_dNqp'], False, None, None, None, None))

        nep = fn.nep_func(self.comb_freq_phase[:, 1], self.comb_freq_phase[:, 0], self.lmfit_nep_props)
        self.nep = nep

        return self.comb_freq_phase[:, 0], nep

    def save_nep(self, phase_nep_fname='phase_nep.dat'):
        freq_nep = np.hstack((self.comb_freq_phase[:, 0].reshape(-1,1), self.nep.reshape(-1,1)))
        np.savetxt(self.save_dir+phase_nep_fname, freq_nep, delimiter=' ', header='freq(Hz) NEP[W/Hz^1/2]')

    def save_Nqp_PS(self, Nqp_PS_fname='oncho_result.dat'):
        T_Terr = np.hstack((self.Tarray.reshape(-1,1), self.Tstart_stop))
        Nqp_PS = np.hstack((self.Nqp.reshape(-1,1), self.phase_shift.reshape(-1,1)))
        T_Terr_Nqp_PS = np.hstack((T_Terr, Nqp_PS))
        np.savetxt(self.save_dir+Nqp_PS_fname, T_Terr_Nqp_PS, delimiter=' ', header='T[mK] T_start[mK] T_stop[mK] Nqp phase_shift[rad]')

    def save_soshi_params(self, save_fname='soshi_props.csv'):
        with open(self.save_dir+save_fname, 'w', newline="") as f  :
            csv_header = ['para_name', 'value', 'sigma']
            csv_rows = [csv_header]
            para_dict = self.lmfit_element_props.valuesdict()
            val_dict = self.lmfit_oncho_result.params.valuesdict()
            sig = np.sqrt(np.diag(self.lmfit_oncho_result.covar))
            for key in para_dict.keys():
                tmp_row = [key, str(para_dict[key]), 'None']
                csv_rows.append(tmp_row)
            for cnt, key in enumerate(val_dict.keys()):
                tmp_row = [key, str(val_dict[key]), str(sig[cnt])]
                csv_rows.append(tmp_row)
            oncho_fit_range_min_row = ['fit_range_min', str(self.oncho_fit_range[0]), 'None']
            oncho_fit_range_max_row = ['fit_range_max', str(self.oncho_fit_range[1]), 'None']

            csv_rows.append(oncho_fit_range_min_row)
            csv_rows.append(oncho_fit_range_max_row)
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)

    def output_soshi_params(self, save_fname='soshi_props.csv'):
        with open(self.save_dir+save_fname, 'w', newline="") as f  :
            csv_header = ['para_name', 'value', 'sigma']
            csv_rows = [csv_header]
            para_dict = self.lmfit_element_props.valuesdict()
            val_dict = self.lmfit_oncho_result.params.valuesdict()
            sig = np.sqrt(np.diag(self.lmfit_oncho_result.covar))
            for key in para_dict.keys():
                tmp_row = [key, str(para_dict[key]), 'None']
                csv_rows.append(tmp_row)
            for cnt, key in enumerate(val_dict.keys()):
                tmp_row = [key, str(val_dict[key]), str(sig[cnt])]
                csv_rows.append(tmp_row)
            oncho_fit_range_min_row = ['fit_range_min', str(self.oncho_fit_range[0]), 'None']
            oncho_fit_range_max_row = ['fit_range_max', str(self.oncho_fit_range[0]), 'None']

            csv_rows.append(oncho_fit_range_min_row)
            csv_rows.append(oncho_fit_range_max_row)
            return csv_rows


class Gaotau(Gao):
    def __init__(self, ref_swp_fname, sg_freq=4000):
        self.trg_fname = "tod_trg.dat"
        self.start_theta = 0.0
        self.lmfit_init_params = Parameters()
        self.lmfit_tau_result = minimizer.MinimizerResult()

        super().__init__(ref_swp_fname, sg_freq)
    
    def get_fr_phase(self):
        tau, xc, yc, r, fr, Qr, phi_0 = self.get_fit_params()
        fr_index = np.argmin(np.abs(self.f-fr))
        xc_c_fr, yc_c_fr = self.set_data_default_position(self.I[fr_index], self.Q[fr_index], self.f[fr_index])
        self.start_theta = np.arctan2(yc_c_fr, xc_c_fr)


    def tod2trg(self, trg_file_name, sample_rate, trg_freq, *fit_params, **kwargs):
        options={"load_fit_file":"none",
                 "loops":False}
        options.update(kwargs)

        trg_tod_data = np.genfromtxt(trg_file_name, delimiter=" ")
        time = trg_tod_data[:, 0]/(sample_rate*1000)
        trg_tod_I = trg_tod_data[:, 1]*sample_rate
        trg_tod_Q = trg_tod_data[:, 2]*sample_rate
        xc_c, yc_c = self.set_data_default_position(trg_tod_I, trg_tod_Q, trg_freq)
        theta = np.arctan2(yc_c, xc_c)

        trg_header_index = np.where(time==0.0)
        cnt = 0
        for i in trg_header_index[0]:
            start = i
            if(cnt>=np.shape(trg_header_index[0])[0]-1):
                #end of loop trg_file
                stop = time.shape[0]+1
                if((stop+1-start)<=1024):
                    break
            else:
                stop = trg_header_index[0][cnt+1]
            one_trg_time = time[start:stop]
            one_trg_phase = self.phase_smoother(theta[start:stop], std_theta=self.start_theta)
            if(cnt==0):
                one_trg = np.hstack((one_trg_time.reshape(-1,1), one_trg_phase.reshape(-1,1)))
                trg_set = np.array([one_trg])
            else:
                one_trg = np.hstack((one_trg_time.reshape(-1,1), one_trg_phase.reshape(-1,1)))
                trg_set = np.append(trg_set, [one_trg], axis=0)
            cnt += 1
        return trg_set

        # if(options["loops"]==False):
        #     # why??
        #     trg_tod_I = trg_tod_data[:, 1]*sample_rate
        #     trg_tod_Q = trg_tod_data[:, 2]*sample_rate

        #     xc_c, yc_c = self.set_data_default_position(trg_tod_I, trg_tod_Q, tod_freq)
        #     theta = np.arctan2(yc_c, xc_c)
        #     phase = self.phase_smoother(theta, **kwargs)
        #     return time, phase

        # elif(options["loops"]==True):
        #     trg_header_index = np.where(time==0.0)
        #     cnt = 0
        #     for i in trg_header_index[0]:
        #         start = trg_header_index[0][cnt]
        #         if(cnt>=np.shape(trg_header_index[0])[0]-1):
        #             stop = time.shape[0]+1
        #         else:
        #             stop = trg_header_index[0][cnt+1]
        #         trg_tod_I = trg_tod_data[start:stop, 1]*sample_rate
        #         trg_tod_Q = trg_tod_data[start:stop, 2]*sample_rate
        #         xc_c, yc_c = self.set_data_default_position(trg_tod_I, trg_tod_Q, tod_freq)
        #         theta = np.arctan2(yc_c, xc_c)
        #         one_trg_time = time[start:stop]
        #         one_trg_phase = self.phase_smoother(theta)
        #         if(cnt==0):
        #             one_trg = np.hstack((one_trg_time.reshape(-1,1), one_trg_phase.reshape(-1,1)))
        #             loop_trg_set = np.array([one_trg])
        #         else:
        #             one_trg = np.hstack((one_trg_time.reshape(-1,1), one_trg_phase.reshape(-1,1)))
        #             loop_trg_set = np.append(loop_trg_set, [one_trg], axis=0)
        #         cnt += 1
        #     return loop_trg_set