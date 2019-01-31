import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from . import gaopy
from . import trgfit
from . import functions as fn

class Kidraw():
    def __init__(self, swp_file_name, sg_freq):
        self.fine_fitting_flag = True
        self.gao_obj = gaopy.Gao(swp_file_name, sg_freq)
        self.save_dir = self.gao_obj.save_dir
        self.tau = 0.0
        self.xc = 0.0
        self.yc = 0.0
        self.r = 0.0
        self.fr = 0.0
        self.Qr = 0.0
        self.phi_0 = 0.0
        self.plt_obj = plt
        self.plt_obj.style.use('default')

    def set_save_dir(self, save_dir='./'):
        self.save_dir = save_dir
        self.gao_obj.set_save_dir(save_dir)

    def get_fit_params(self, **kwargs):
        options = {'avoid_fine_fit':False}
        options.update(kwargs)
        coarse_fit_params = self.gao_obj.coarse_fit(**kwargs)
        try:
            if(options['avoid_fine_fit']==True):
                raise ValueError("avoid fine fitting")
            else:
                pass
            fine_fit_params = self.gao_obj.fine_fit(**kwargs)
            fit_params = fine_fit_params
            self.fine_fitting_flag = False
        except AttributeError:
            print("fine fitting : Failed")
            fit_params = coarse_fit_params
        self.tau = fit_params[0]
        self.xc = fit_params[1]
        self.yc = fit_params[2]
        self.r = fit_params[3]
        self.fr = fit_params[4]
        self.Qr = fit_params[5]
        self.phi_0 = fit_params[6]

    def plot_sweep(self, **kwargs):
        options = {'save':False}
        options.update(kwargs)

        dpi_val = 200
        x, y = self.gao_obj.remove_tau_effect(self.gao_obj.I, self.gao_obj.Q, self.gao_obj.f, self.tau)
        xc_c, yc_c = self.gao_obj.set_data_default_position(self.gao_obj.I, self.gao_obj.Q, self.gao_obj.f)
        
        crs_x, crs_y = self.gao_obj.remove_tau_effect(self.gao_obj.I, self.gao_obj.Q, self.gao_obj.f, self.gao_obj.lmfit_init_params['tau']) 
        crs_xc_c, crs_yc_c = self.gao_obj.set_data_default_position(crs_x, crs_y, self.gao_obj.f, coarse_fit=True, xc=self.gao_obj.lmfit_init_circle_params['xc'], yc=self.gao_obj.lmfit_init_circle_params['yc'])
        crs_theta = np.arctan2(crs_yc_c, crs_xc_c)
        crs_theta_c = self.gao_obj.phase_smoother(crs_theta)

        amp_dB = 10*np.log10(np.sqrt(self.gao_obj.I**2+self.gao_obj.Q**2))

        phase = np.arctan2(self.gao_obj.Q, self.gao_obj.I)
        smooth_phase = self.gao_obj.phase_smoother(phase)

        fig_IQ = self.plt_obj.figure('IQ')
        ax_IQ = fig_IQ.add_subplot(111)
        ax_IQ.tick_params(direction='in')
        ax_IQ.grid(True)
        ax_IQ.set_title("sweep result")
        ax_IQ.set_xlabel("I")
        ax_IQ.set_ylabel("Q")
        ax_IQ.set_aspect('equal', 'datalim')
        ax_IQ.scatter(self.gao_obj.I, self.gao_obj.Q, label="I/Q", color='r')
        ax_IQ.scatter(x, y, label="I/Q (removed tau)", color='b')
        ax_IQ.scatter(xc_c, yc_c, label="I/Q (centerd)", color='g')
        ax_IQ.scatter(self.xc, self.yc, label='circle center', color='k')

        if(self.fine_fitting_flag==False):
            fit_f = np.linspace(self.gao_obj.fine_fit_range[0], self.gao_obj.fine_fit_range[1], 100)
            IQ = fn.t21_func(fit_f, self.gao_obj.lmfit_result.params)
            fit_I = np.real(IQ)
            fit_Q = np.imag(IQ)
            fit_x, fit_y = self.gao_obj.remove_tau_effect(fit_I, fit_Q, fit_f, self.tau)
            fit_xc_c, fit_yc_c = self.gao_obj.set_data_default_position(fit_I, fit_Q, fit_f)
            ax_IQ.plot(fit_I, fit_Q, label="fit_I/Q", color='y')
            ax_IQ.plot(fit_x, fit_y, label="fit_I/Q (removed tau)", color='y', linestyle=':')
            ax_IQ.plot(fit_xc_c, fit_yc_c, label="fit_I/Q (centered)", color='y', linestyle='-.')
        else:
            pass

        self.plt_obj.legend(loc='upper right')
        self.plt_obj.gca().set_aspect('equal', adjustable='box')

        fig_theta_func = self.plt_obj.figure('theta_fit')
        start_theta=-self.gao_obj.lmfit_theta_result.params['theta_0']
        theta_func_fit_index = np.where((self.gao_obj.f>self.gao_obj.theta_fit_range[0])&(self.gao_obj.f<self.gao_obj.theta_fit_range[1]))
        theta_func_plot_f = np.linspace(self.gao_obj.theta_fit_range[0], self.gao_obj.theta_fit_range[1], 100)
        theta_func_plot_p = fn.theta_func(theta_func_plot_f, self.gao_obj.lmfit_theta_result.params)
        ax_theta_func = fig_theta_func.add_subplot(111)
        ax_theta_func.tick_params(direction='in')
        ax_theta_func.grid(True)
        ax_theta_func.set_title("sweep result")
        ax_theta_func.set_xlabel("frequancy[MHz]")
        ax_theta_func.set_ylabel("theta[rad]")
        ax_theta_func.scatter(self.gao_obj.f[theta_func_fit_index]/1e6, crs_theta_c[theta_func_fit_index]-start_theta, label='data')
        ax_theta_func.plot(theta_func_plot_f/1e6, theta_func_plot_p-start_theta, label='fitting', color='crimson')
        self.plt_obj.legend(loc='upper right')

        fig_all = self.plt_obj.figure('swp_all')
        ax_fI = fig_all.add_subplot(411)
        ax_fI.grid(True)
        ax_fI.set_xlabel("Frequancy[MHz]")
        ax_fI.set_ylabel("I")
        ax_fI.plot(self.gao_obj.f/1e6, self.gao_obj.I, label="freq vs I", color='k')

        ax_fQ = fig_all.add_subplot(412)
        ax_fQ.grid(True)
        ax_fQ.set_xlabel("Frequancy[MHz]")
        ax_fQ.set_ylabel("Q")
        ax_fQ.plot(self.gao_obj.f/1e6, self.gao_obj.Q, label="freq vs Q", color='k')
        
        ax_fa = fig_all.add_subplot(413)
        ax_fa.grid(True)
        ax_fa.set_xlabel("Frequancy[MHz]")
        ax_fa.set_ylabel("Amplitude[dB]")
        ax_fa.plot(self.gao_obj.f/1e6, amp_dB, label="freq vs amplitude", color='k')

        ax_fp = fig_all.add_subplot(414)
        ax_fp.grid(True)
        ax_fp.set_xlabel("Frequancy[MHz]")
        ax_fp.set_ylabel("Phase[rad]")
        ax_fp.plot(self.gao_obj.f/1e6, smooth_phase, label="freq vs phase", color='k')
        fig_all.tight_layout()

        if(options['save']==True):
                save_fig = self.plt_obj.figure('IQ')
                save_fig.suptitle('IQ plot')
                save_fig.savefig(self.save_dir+'sweep_IQ.pdf', dpi=200)

                save_fig = self.plt_obj.figure('theta_fit')
                save_fig.suptitle('theta fit')
                save_fig.savefig(self.save_dir+'theta_fit.pdf', dpi=200)

                save_fig = self.plt_obj.figure('swp_all')
                save_fig.suptitle('all sweep results')
                save_fig.savefig(self.save_dir+'all_sweep.pdf', dpi=200)

        
    def check_tod(self, tod_freq, tod_file_list=[["tod_1ksps.dat", 1.0], ["tod_100ksps.dat", 100.0], ["tod_1Msps.dat", 1000.0]], **kwargs):
        options = {'save':False}
        options.update(kwargs)
        
        x, y = self.gao_obj.remove_tau_effect(self.gao_obj.I, self.gao_obj.Q, self.gao_obj.f, self.tau)
        xc_c, yc_c = self.gao_obj.set_data_default_position(self.gao_obj.I, self.gao_obj.Q, self.gao_obj.f)
        for tod_file in tod_file_list:
            tod_data = np.genfromtxt(tod_file[0], delimiter=" ")
            tod_I = tod_data[:, 1]*tod_file[1]
            tod_Q = tod_data[:, 2]*tod_file[1]
            tod_x, tod_y = self.gao_obj.remove_tau_effect(tod_I, tod_Q, tod_freq, self.tau)
            tod_xc_c, tod_yc_c = self.gao_obj.set_data_default_position(tod_I, tod_Q, tod_freq)

            fig_size=(6,6)
            dpi_val = 200
            fig = self.plt_obj.figure(tod_file[0].split('/')[-1])
            ax = fig.add_subplot(111)
            ax.tick_params(direction='in')
            ax.grid(True)
            ax.set_title("sweep result")
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.scatter(self.gao_obj.I, self.gao_obj.Q, label="I/Q", color='r')
            ax.scatter(x, y, label="I/Q (removed tau)", color='b')
            ax.scatter(xc_c, yc_c, label="I/Q (centerd)", color='g')
            ax.scatter(self.xc, self.yc, label='circle center', color='k')
            ax.scatter(tod_I, tod_Q, color='w', edgecolors='k', marker='o')
            ax.scatter(tod_x, tod_y, color='w', edgecolors='k', marker='o')
            ax.scatter(tod_xc_c, tod_yc_c, color='w', edgecolors='k', marker='o')
            self.plt_obj.legend()
            self.plt_obj.gca().set_aspect('equal', adjustable='box')

        if(options['save']==True):
            save_fig = self.plt_obj.figure('tod_1ksps.dat')
            save_fig.suptitle('tod plot (1ksps)')
            save_fig.savefig(self.save_dir+'tod_1ksps.png', dpi=dpi_val)

            save_fig = self.plt_obj.figure('tod_100ksps.dat')
            save_fig.suptitle('tod plot (100ksps)')
            save_fig.savefig(self.save_dir+'tod_100ksps.png', dpi=dpi_val)

            save_fig = self.plt_obj.figure('tod_1Msps.dat')
            save_fig.suptitle('tod plot (1Msps)')
            save_fig.savefig(self.save_dir+'tod_1Msps.png', dpi=dpi_val)

    def draw_psd(self,  tod_freq, tod_file_list, **kwargs):
        options = {'save':False}
        options.update(kwargs)

        #psd_unit option is efective to only return array. menber variable's unit is still liner
        comb_freq_amp_welch, comb_freq_phase_welch = self.gao_obj.comb_psd(tod_freq, tod_file_list, psd_unit='dB', **kwargs)
        comb_freq_amp_peri, comb_freq_phase_peri = self.gao_obj.comb_psd(tod_freq, tod_file_list, psd_method = "periodogram", psd_unit='dB', save_to_menber=False, **kwargs)
        
        fig_amp = self.plt_obj.figure('amp_psd')
        ax_amp = fig_amp.add_subplot(111)
        ax_amp.tick_params(direction='in')
        ax_amp.grid(True)
        ax_amp.set_title("amplitude PSD result")
        ax_amp.set_xlabel("Frequancy [Hz]")
        ax_amp.set_ylabel("amp PSD [dBc/Hz]")
        ax_amp.set_xscale("log", nonposx = 'clip')
        ax_amp.plot(comb_freq_amp_peri[:,0], comb_freq_amp_peri[:,1], label='periodogram', color='k')
        ax_amp.plot(comb_freq_amp_welch[:,0], comb_freq_amp_welch[:,1], label='welch', color='r')
        self.plt_obj.legend()

        fig_phase = self.plt_obj.figure('phase_psd')
        ax_phase = fig_phase.add_subplot(111)
        ax_phase.tick_params(direction='in')
        ax_phase.grid(True)
        ax_phase.set_title("phase PSD result")
        ax_phase.set_xlabel("Frequancy [Hz]")
        ax_phase.set_ylabel("phase PSD [dBc/Hz]")
        ax_phase.set_xscale("log", nonposx = 'clip')
        ax_phase.plot(comb_freq_phase_peri[:,0], comb_freq_phase_peri[:,1], label='periodogram', color='k')
        ax_phase.plot(comb_freq_phase_welch[:,0], comb_freq_phase_welch[:,1], label='welch', color='b')
        self.plt_obj.legend()

        fig_avp = self.plt_obj.figure('amp_vs_phase')
        ax_avp = fig_avp.add_subplot(111)
        ax_avp.tick_params(direction='in')
        ax_avp.grid(True)
        ax_avp.set_title("amplitude PSD and phase PSD")
        ax_avp.set_xlabel("Frequancy [Hz]")
        ax_avp.set_ylabel("PSD [dBc/Hz]")
        ax_avp.set_xscale("log", nonposx = 'clip')
        ax_avp.plot(comb_freq_phase_welch[:,0], comb_freq_phase_welch[:,1], label='phase', color='b')
        ax_avp.plot(comb_freq_amp_welch[:,0], comb_freq_amp_welch[:,1], label='amplitude', color='r')
        self.plt_obj.legend()

        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                save_fig.savefig(self.save_dir+fig_lb+'.pdf', dpi=200)

    def close_plt_obj(self):
        self.plt_obj.close()




class Nepdraw(Kidraw):
    def __init__(self, ref_swp_fname, sg_freq, **kwargs):
        super().__init__(ref_swp_fname, sg_freq, **kwargs)
        self.gao_obj = gaopy.Gaonep(ref_swp_fname, sg_freq)
        self.oncho_file_list=[]

    def draw_oncho(self, oncho_file_list, **kwargs):
        options = {'save':False}
        options.update(kwargs)

        self.oncho_file_list=oncho_file_list
        fig = self.plt_obj.figure('oncho')
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_title("oncho result")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect('equal', 'datalim')
        ax.scatter(0, 0, label='center', color='k', marker='+')

        cnt = 0
        Tarray = []
        fname_list = []
        segments = []
        xlims = np.array([])
        ylims = np.array([])
        for ele in self.oncho_file_list:
            fname_list.append(ele[0])
            Tarray = np.append(Tarray, ele[1])
        Tmin = np.min(Tarray).reshape((1,-1))[0][0]
        Tmax = np.max(Tarray).reshape((1,-1))[0][0]

        for fname in fname_list:
            swp_data = np.genfromtxt(fname, delimiter=" ")
            tmp_I = swp_data[:,1]
            tmp_Q = swp_data[:,2]

            tmp_f = self.gao_obj.sg*1.0E6 + swp_data[:,0]
            tmp_fr_idx = np.abs(tmp_f-self.fr).argmin()
            oncho_xc_c, oncho_yc_c = self.gao_obj.set_data_default_position(tmp_I, tmp_Q, tmp_f)
            segments.append(list(zip(oncho_xc_c, oncho_yc_c)))

            if(cnt==0):
                ax.scatter(oncho_xc_c[tmp_fr_idx], oncho_yc_c[tmp_fr_idx], label='fr point',color='w', edgecolor='k', zorder=2, s=8)
                xlims = [[np.min(oncho_xc_c), np.max(oncho_xc_c)]]
                ylims = [[np.min(oncho_yc_c), np.max(oncho_yc_c)]]
            else:
                ax.scatter(oncho_xc_c[tmp_fr_idx], oncho_yc_c[tmp_fr_idx], color='w', edgecolor='k', zorder=2, s=8)
                xlims = np.append(xlims, [[np.min(oncho_xc_c), np.max(oncho_xc_c)]], axis=0)
                ylims = np.append(ylims, [[np.min(oncho_yc_c), np.max(oncho_yc_c)]], axis=0)
            #ax.plot(oncho_xc_c, oncho_yc_c, marker='o', markersize=2, zorder=1, color=cm.inferno((Tarray[cnt]-Tmin)/(Tmax-Tmin)))
            cnt += 1
        expand = 1.05
        ax.set_xlim(xlims[:, 0].min()*expand, xlims[:, 1].max()*expand)
        ax.set_ylim(ylims[:, 0].min()*expand, ylims[:, 1].max()*expand)
        norm = colors.Normalize(Tmin, Tmax)
        line_segments = LineCollection(segments, linewidths=2, cmap='inferno', norm=norm, zorder=1)
        line_segments.set_array(Tarray)
        ax.add_collection(line_segments)
        axcb = fig.colorbar(line_segments)
        axcb.set_label("Temperature (mK)")
        self.plt_obj.gca().set_aspect('equal', adjustable='box')

        if(options['save']==True):
            save_fig = self.plt_obj.figure('oncho')
            save_fig.savefig(self.save_dir+'oncho.pdf', dpi=200)
            

    def plot_phase_shift(self, oncho_file_list, N0, delta_0, volume, **kwargs):
        options = {'save':False}
        options.update(kwargs)

        self.gao_obj.oncho_analisys(oncho_file_list, self.fr, self.Qr)
        self.gao_obj.Temp2Nqp(N0, delta_0, volume)

        fig_TvsPS = self.plt_obj.figure('T_vs_PS')
        ax_TvsPS = fig_TvsPS.add_subplot(111)
        ax_TvsPS.tick_params(direction='in')
        ax_TvsPS.set_title("Temperature vs Phase Shift")
        ax_TvsPS.set_ylabel("Phase Shift (rad)")
        ax_TvsPS.set_xlabel("Tempreture (mK)")
        #ax_TvsPS.scatter(self.gao_obj.Tarray, self.gao_obj.phase_shift)
        low_xerr = self.gao_obj.Tarray -self.gao_obj.Tstart_stop[:, 0]
        upp_xerr = self.gao_obj.Tstart_stop[:, 1]-self.gao_obj.Tarray
        ax_TvsPS.errorbar(self.gao_obj.Tarray, self.gao_obj.phase_shift, xerr=[low_xerr, upp_xerr], fmt='o')

        fig_NqpvsPS = self.plt_obj.figure('Nqp_vs_PS')
        ax_NqpvsPS = fig_NqpvsPS.add_subplot(111)
        ax_NqpvsPS.tick_params(direction='in')
        ax_NqpvsPS.set_title("Nqp vs Phase Shift")
        ax_NqpvsPS.set_ylabel("Phase Shift (rad)")
        ax_NqpvsPS.set_xlabel("Nqp")
        ax_NqpvsPS.scatter(self.gao_obj.Nqp, self.gao_obj.phase_shift)

        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                save_fig.savefig(self.save_dir+fig_lb+'.pdf', dpi=200)

    def load_psd(self, amp_psd_fname, phase_psd_fname, skip_header=1):
        self.gao_obj.load_psd(amp_psd_fname, phase_psd_fname, skip_header=skip_header)

    def get_psd(self,  tod_freq, tod_file_list, **kwargs):
        options={"load_fit_file":"none",
                 "psd_unit":"liner",
                 "psd_method":"welch",
                 "segment_rate":50,
                 "ps_option":False}
        options.update(kwargs)
        self.gao_obj.get_psd(tod_freq, tod_file_list, **options)


    def plot_nep(self, delta, eta, tau_qp, fit_Nqp_min, fit_Nqp_max, init_dth_dNqp=1.0, init_phase_bias=0.0, **kwargs):
        options = {'save':False}
        options.update(kwargs)
        
        freq, nep = self.gao_obj.calc_nep(delta, eta, tau_qp, fit_Nqp_min, fit_Nqp_max, init_dth_dNqp, init_phase_bias)
        
        fig_NqpvsPSFit = self.plt_obj.figure('Nqp_vs_PS+Fit', figsize=(6.4, 4.8), dpi=200)
        ax_NqpvsPSFit = fig_NqpvsPSFit.add_subplot(111)
        ax_NqpvsPSFit.tick_params(direction='in')
        ax_NqpvsPSFit.set_title("Nqp vs Phase Shift")
        ax_NqpvsPSFit.set_ylabel("Phase Shift (rad)")
        ax_NqpvsPSFit.set_xlabel("Nqp")
        ax_NqpvsPSFit.scatter(self.gao_obj.Nqp, self.gao_obj.phase_shift)
        fit_Nqp = np.linspace(fit_Nqp_min, fit_Nqp_max)
        ax_NqpvsPSFit.plot(fit_Nqp, fn.phase_Nqp_func(fit_Nqp, self.gao_obj.lmfit_oncho_result.params), label='fitting', color='r')
        self.plt_obj.legend()

        fig_nep = self.plt_obj.figure('NEP', figsize=(6.4, 4.8), dpi=200)
        ax_nep = fig_nep.add_subplot(111)
        ax_nep.tick_params(direction='in')
        ax_nep.set_title("NEP")
        ax_nep.set_xlabel("Frequancy [Hz]")
        ax_nep.set_ylabel("NEP [W/Hz^1/2]")
        ax_nep.set_xscale('log', nonposx = 'clip')
        ax_nep.set_yscale('log', nonposy = 'clip')
        ax_nep.plot(freq, nep)

        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                save_fig.savefig(self.save_dir+fig_lb+'.pdf', dpi=200)



class Taudraw():
    def __init__(self,  trg_swp_file_list = [['tod_trg.dat', 'sweep.dat', 4000, 1000, 4011.12e6]], **kwargs):
        self.trg_swp_file_list = trg_swp_file_list
        self.trg_spr_freq_dict = {}
        self.gao_obj_dict = {}
        self.fit_info = pd.DataFrame([])
        self.swp_params_data = pd.DataFrame([])
        self.save_dir = './'
        self.plt_obj = plt
        self.plt_obj.style.use('default')
        for trg_swp_file in trg_swp_file_list:
            trg_fname = trg_swp_file[0]
            trg_swp_fname = trg_swp_file[1]
            sg_freq = trg_swp_file[2]
            sample_rate = trg_swp_file[3]
            trg_freq = trg_swp_file[4]
            gao_obj = gaopy.Gaotau(trg_swp_fname, sg_freq)
            self.gao_obj_dict[trg_fname] = gao_obj
            self.trg_spr_freq_dict[trg_fname] = [sample_rate, trg_freq]
        self.trg_file_dict = {}
        self.combined_df = pd.DataFrame([])

    def set_save_dir(self, save_dir='./'):
        self.save_dir = save_dir
        for gao_obj in self.gao_obj_dict.keys():
            self.gao_obj_dict[gao_obj].set_save_dir(save_dir)

    def get_fit_params(self, **kwargs):
        options = {'avoid_fine_fit_dict':[]}
        options.update(kwargs)
        swp_fit_params_header = ['tau', 'xc', 'yc', 'r', 'fr', 'Qr', 'phi_0', 'fine_fit_success']
        swp_fit_params_index = []
        swp_fit_params_list = [swp_fit_params_header]
        for trg_file in self.gao_obj_dict.keys():
            coarse_fit_params = self.gao_obj_dict[trg_file].coarse_fit(**kwargs)
            try:
                for avoid in options['avoid_fine_fit_dict'] :
                    if(trg_file == avoid):
                        raise ValueError()

                fine_fit_params = self.gao_obj_dict[trg_file].fine_fit(**kwargs)
                if((self.gao_obj_dict[trg_file].lmfit_result.ier in (1, 2, 3))==False):
                    print("fine fitting : Failed")
                    fit_params = coarse_fit_params
                elif(self.gao_obj_dict[trg_file].lmfit_result.ier in (1, 2, 3)):
                    fit_params = fine_fit_params
                fine_fit_success = self.gao_obj_dict[trg_file].lmfit_result.success
            except ValueError:
                print("fine fitting : Avoid")
                fit_params = coarse_fit_params
                fine_fit_success = False
            fit_results = np.append(fit_params, fine_fit_success)
            swp_fit_params_list.append(fit_results)
            swp_fit_params_index.append(trg_file.split('/')[-2]+'/'+trg_file.split('/')[-1])
        self.swp_params_data = pd.DataFrame(swp_fit_params_list[1:], index = swp_fit_params_index, columns=swp_fit_params_header)
        print(self.swp_params_data)


    def load_trg(self):
        header = ['file_name', 'succeeded', 'failed', 'total']
        fit_info_list = [header]
        for trg_file in self.trg_spr_freq_dict.keys():
            print("loading" + trg_file + "...")
            trg_set = self.gao_obj_dict[trg_file].tod2trg(trg_file, self.trg_spr_freq_dict[trg_file][0], self.trg_spr_freq_dict[trg_file][1])
            trgholder = trgfit.Trgholder(self.gao_obj_dict[trg_file].swp_file_name, trg_file, self.trg_spr_freq_dict[trg_file][0])
            trgholder.analyze_trg(trg_set)
            self.trg_file_dict[trg_file] = trgholder
            tmp_row = [trg_file.split('/')[-2]+'/'+trg_file.split('/')[-1], len(trgholder.oneshot_list), len(trgholder.failed_list), len(trgholder.oneshot_list)+len(trgholder.failed_list)]
            fit_info_list.append(tmp_row)
        fit_info = pd.DataFrame(fit_info_list[1:], columns=header)
        self.fit_info = fit_info.set_index('file_name')
        self.combined_df = pd.concat([trgholder.analyzed_data for trgholder in self.trg_file_dict.values()], keys=[trg_fname.split('/')[-2]+'/'+trg_fname.split('/')[-1] for trg_fname in self.trg_file_dict.keys()])
        pd.options.display.float_format = '{:.2e}'.format

    def plot_sweep(self, **kwargs):
        options = {'save':False,
        'trg_fname':self.trg_swp_file_list[0][0]}
        options.update(kwargs)

        dpi_val = 200
        plot_I = self.gao_obj_dict[options['trg_fname']].I
        plot_Q = self.gao_obj_dict[options['trg_fname']].Q
        plot_f = self.gao_obj_dict[options['trg_fname']].f
        plot_tau, plot_xc, plot_yc, plot_fine_fit_success = self.swp_params_data.loc[options['trg_fname'].split('/')[-2]+'/'+options['trg_fname'].split('/')[-1], ['tau', 'xc', 'yc', 'fine_fit_success']]

        x, y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(plot_I, plot_Q, plot_f, plot_tau)
        xc_c, yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(plot_I, plot_Q, plot_f)
        
        crs_x, crs_y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(plot_I, plot_Q, plot_f, self.gao_obj_dict[options['trg_fname']].lmfit_init_params['tau']) 
        crs_xc_c, crs_yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(crs_x, crs_y, plot_f, coarse_fit=True, xc=self.gao_obj_dict[options['trg_fname']].lmfit_init_circle_params['xc'], yc=self.gao_obj_dict[options['trg_fname']].lmfit_init_circle_params['yc'])
        crs_theta = np.arctan2(crs_yc_c, crs_xc_c)
        crs_theta_c = self.gao_obj_dict[options['trg_fname']].phase_smoother(crs_theta)

        amp_dB = 10*np.log10(np.sqrt(plot_I**2+plot_Q**2))

        phase = np.arctan2(plot_Q, plot_I)
        smooth_phase = self.gao_obj_dict[options['trg_fname']].phase_smoother(phase)

        fig_IQ = self.plt_obj.figure('IQ')
        ax_IQ = fig_IQ.add_subplot(111)
        ax_IQ.tick_params(direction='in')
        ax_IQ.grid(True)
        ax_IQ.set_title("sweep result")
        ax_IQ.set_xlabel("I")
        ax_IQ.set_ylabel("Q")
        ax_IQ.set_aspect('equal', 'datalim')
        ax_IQ.scatter(plot_I, plot_Q, label="I/Q", color='r')
        ax_IQ.scatter(x, y, label="I/Q (removed tau)", color='b')
        ax_IQ.scatter(xc_c, yc_c, label="I/Q (centerd)", color='g')
        ax_IQ.scatter(plot_xc, plot_yc, label='circle center', color='k')

        if(plot_fine_fit_success==True):
            fit_f = np.linspace(self.gao_obj_dict[options['trg_fname']].fine_fit_range[0], self.gao_obj_dict[options['trg_fname']].fine_fit_range[1], 100)
            IQ = fn.t21_func(fit_f, self.gao_obj_dict[options['trg_fname']].lmfit_result.params)
            fit_I = np.real(IQ)
            fit_Q = np.imag(IQ)
            fit_x, fit_y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(fit_I, fit_Q, fit_f, plot_tau)
            fit_xc_c, fit_yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(fit_I, fit_Q, fit_f)
            ax_IQ.plot(fit_I, fit_Q, label="fit_I/Q", color='y')
            ax_IQ.plot(fit_x, fit_y, label="fit_I/Q (removed tau)", color='y', linestyle=':')
            ax_IQ.plot(fit_xc_c, fit_yc_c, label="fit_I/Q (centered)", color='y', linestyle='-.')
        else:
            pass

        self.plt_obj.legend(loc='upper right')
        self.plt_obj.gca().set_aspect('equal', adjustable='box')

        fig_theta_func = self.plt_obj.figure('theta_fit')
        start_theta=-self.gao_obj_dict[options['trg_fname']].lmfit_theta_result.params['theta_0']
        theta_func_fit_index = np.where((plot_f>self.gao_obj_dict[options['trg_fname']].theta_fit_range[0])&(plot_f<self.gao_obj_dict[options['trg_fname']].theta_fit_range[1]))
        theta_func_plot_f = np.linspace(self.gao_obj_dict[options['trg_fname']].theta_fit_range[0], self.gao_obj_dict[options['trg_fname']].theta_fit_range[1], 100)
        theta_func_plot_p = fn.theta_func(theta_func_plot_f, self.gao_obj_dict[options['trg_fname']].lmfit_theta_result.params)
        ax_theta_func = fig_theta_func.add_subplot(111)
        ax_theta_func.tick_params(direction='in')
        ax_theta_func.grid(True)
        ax_theta_func.set_title("sweep result")
        ax_theta_func.set_xlabel("frequancy[MHz]")
        ax_theta_func.set_ylabel("theta[rad]")
        ax_theta_func.scatter(plot_f[theta_func_fit_index]/1e6, crs_theta_c[theta_func_fit_index]-start_theta, label='data')
        ax_theta_func.plot(theta_func_plot_f/1e6, theta_func_plot_p-start_theta, label='fitting', color='crimson')
        self.plt_obj.legend(loc='upper right')

        fig_all = self.plt_obj.figure('swp_all')
        ax_fI = fig_all.add_subplot(411)
        ax_fI.grid(True)
        ax_fI.set_xlabel("Frequancy[MHz]")
        ax_fI.set_ylabel("I")
        ax_fI.plot(plot_f/1e6, plot_I, label="freq vs I", color='k')

        ax_fQ = fig_all.add_subplot(412)
        ax_fQ.grid(True)
        ax_fQ.set_xlabel("Frequancy[MHz]")
        ax_fQ.set_ylabel("Q")
        ax_fQ.plot(plot_f/1e6, plot_Q, label="freq vs Q", color='k')
        
        ax_fa = fig_all.add_subplot(413)
        ax_fa.grid(True)
        ax_fa.set_xlabel("Frequancy[MHz]")
        ax_fa.set_ylabel("Amplitude[dB]")
        ax_fa.plot(plot_f/1e6, amp_dB, label="freq vs amplitude", color='k')

        ax_fp = fig_all.add_subplot(414)
        ax_fp.grid(True)
        ax_fp.set_xlabel("Frequancy[MHz]")
        ax_fp.set_ylabel("Phase[rad]")
        ax_fp.plot(plot_f/1e6, smooth_phase, label="freq vs phase", color='k')
        fig_all.tight_layout()

        if(options['save']==True):
                save_fig = self.plt_obj.figure('IQ')
                save_fig.suptitle('IQ plot')
                save_fig.savefig(self.save_dir+'sweep_IQ.pdf', dpi=200)

                save_fig = self.plt_obj.figure('theta_fit')
                save_fig.suptitle('theta fit')
                save_fig.savefig(self.save_dir+'theta_fit.pdf', dpi=200)

                save_fig = self.plt_obj.figure('swp_all')
                save_fig.suptitle('all sweep results')
                save_fig.savefig(self.save_dir+'all_sweep.pdf', dpi=200)

    def plot_trg(self, **kwargs):
        options = {'save':False,
        'trg_index':0,
        'trg_fname':self.trg_swp_file_list[0][0],
        'noise_plot':False}
        options.update(kwargs)
        trg_fig = self.plt_obj.figure('one_trg')
        trg_ax = trg_fig.add_subplot(111)
        trg_ax.set_title('one trigger waveform')
        trg_ax.set_xlabel('time [$\\mu s$]')
        trg_ax.set_ylabel('phase [rad]')
        trg_ax.grid(True, zorder=0)

        plot_trgholder = self.trg_file_dict[options['trg_fname']]
        if(options['noise_plot']==True):
            if(len(plot_trgholder.failed_list)>=(options['trg_index']+1)):
                time, phase = plot_trgholder.failed_list[options['trg_index']].output_data()
            else:
                print("No trgger waveform was found.")
                print("plot first failed trigger waveform")
                time, phase = plot_trgholder.failed_list[0].output_data()
        elif(options['noise_plot']==False):
            if(len(plot_trgholder.oneshot_list)>=(options['trg_index']+1)):
                time, phase = plot_trgholder.oneshot_list[options['trg_index']].output_data()
                fit_time_min, fit_time_max = plot_trgholder.oneshot_list[options['trg_index']].phase_fit_range
                fit_time = np.linspace(fit_time_min, fit_time_max)
                fit_phase = fn.phase_tau_func(fit_time, plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params)
                params_row = plot_trgholder.analyzed_data.loc[options['trg_index'],:]
                trg_ax.plot(fit_time*1e6, fit_phase, color='r', label='fit: $\\tau$ = {0:.2e} $\\mu s$'.format(plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params.valuesdict()['phase_tau']*1e6), zorder=10)
            else:
                print("No trgger waveform was found.")
                print("plot first failed trigger waveform")
                time, phase = plot_trgholder.failed_list[0].output_data()

        trg_ax.plot(time*1e6, phase, zorder=5)
        
        self.plt_obj.legend()
        if(options['save']==True):
            save_fig = self.plt_obj.figure('one_trg')
            save_fig.savefig(self.save_dir+'one_trigger_waveform.pdf', dpi=200)

        if(options['noise_plot']==False):
            return params_row

    def plot_histogram(self, **kwargs):
        options = {'tau_bins':50,
        'tau_min':0,
        'tau_max':1000,
        'amp_bins':200,
        'amp_min':-0.5,
        'amp_max':0.5}
        options.update(kwargs)

        fig_tau = self.plt_obj.figure('tau_hist')
        ax_tau = fig_tau.add_subplot(111)
        ax_tau.set_title('$\\tau$ histogram')
        ax_tau.set_xlabel('$\\tau$ [$\\mu s$]')
        ax_tau.set_ylabel('events')
        ax_tau.grid(True, zorder=0)
        tau_n, tau_bins, tau_patches = ax_tau.hist(self.combined_df['phase_tau']*1e6, bins=options['tau_bins'], range=(options['tau_min'], options['tau_max']), zorder=5)

        fig_amp = self.plt_obj.figure('amp_hist')
        ax_amp = fig_amp.add_subplot(111)
        ax_amp.set_title('Amp. histogram')
        ax_amp.set_xlabel('Amp. [rad]')
        ax_amp.set_ylabel('events')
        ax_amp.grid(True, zorder=0)
        amp_n, amp_bins, amp_patches = ax_amp.hist(self.combined_df['phase_Amp'], bins=options['amp_bins'], range=(options['amp_min'], options['amp_max']), zorder=5)



    

