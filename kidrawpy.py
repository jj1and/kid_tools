import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from lmfit.models import LognormalModel
from . import gaopy
from . import trgfit
from . import functions as fn

class Kidraw():
    def __init__(self, swp_file_name, sg_freq):
        self.fine_fit_success = False
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
                self.gao_obj.fine_fit(**kwargs)
                if(self.gao_obj.lmfit_result.redchi<self.gao_obj.fine_fit_redchi_ther):
                    self.fine_fit_success = True
                elif(self.gao_obj.lmfit_result.redchi>=self.gao_obj.fine_fit_redchi_ther):
                    self.fine_fit_success = False
                self.tau, self.xc, self.yc, self.r, self.fr, self.Qr, self.phi_0 = self.gao_obj.get_fit_params()
        except ValueError:
            print("fine fitting : avoided")
            self.tau, self.xc, self.yc, self.r, self.fr, self.Qr, self.phi_0 = coarse_fit_params
            self.fine_fit_success = False

        return [self.tau, self.xc, self.yc, self.r, self.fr, self.Qr, self.phi_0]

    def plot_sweep(self, **kwargs):
        options = {'save':False,
        'loc':'upper right'}
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

        fr_index = np.argmin(np.abs(self.fr-self.gao_obj.f))

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
        ax_IQ.scatter(xc_c[fr_index], yc_c[fr_index], label='fr', color='w', edgecolors='k', marker='^')

        if(self.fine_fit_success == True):
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

        self.plt_obj.legend(loc=options['loc'])
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
        self.plt_obj.legend(loc=options['loc'])

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
        options = {
        'save':False,
        'avoid_fine_fit':False}
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

        fig2 = self.plt_obj.figure('oncho_amp')
        ax2 = fig2.add_subplot(111)
        ax2.grid(True)
        ax2.set_title("oncho result(Amp.)")
        ax2.set_xlabel("freq [MHz]")
        ax2.set_ylabel("Amp. [dB]")

        fig3 = self.plt_obj.figure('oncho_phase')
        ax3 = fig3.add_subplot(111)
        ax3.grid(True)
        ax3.set_title("oncho result(Phase)")
        ax3.set_xlabel("freq [MHz]")
        ax3.set_ylabel("Phase [rad]")

        cnt = 0
        Tarray = []
        fname_list = []
        segments = []
        amp_segments = []
        phase_segments = []
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
            tmp_amp = np.sqrt(tmp_I**2 + tmp_Q**2)
            tmp_log_amp = 10*np.log10(tmp_amp/tmp_amp[0])

            tmp_f = self.gao_obj.sg*1.0E6 + swp_data[:,0]
            tmp_fr_idx = np.abs(tmp_f-self.fr).argmin()
            oncho_xc_c, oncho_yc_c = self.gao_obj.set_data_default_position(tmp_I, tmp_Q, tmp_f)
            tmp_phase = np.arctan2(oncho_yc_c, oncho_xc_c)
            if(cnt==0):
                std_theta = tmp_phase[tmp_fr_idx]
            tmp_phase_c = self.gao_obj.phase_smoother(tmp_phase, std_theta=std_theta)

            # tmp_gao = gaopy.Gaonep(fname, self.gao_obj.sg)
            # tmp_gao.coarse_fit()
            # if(options['avoid_fine_fit']==False):
            #     tmp_gao.fine_fit(**kwargs)
            # elif(options['avoid_fine_fit']==True):
            #     print("avoid fine fitting!!")
            # tau, xc, yc, r, fr, Qr, phi_0 = tmp_gao.get_fit_params()
            # tmp_fr_idx = np.abs(tmp_gao.f-self.fr).argmin()
            # fr_idx = np.abs(tmp_gao.f-fr).argmin()
            # oncho_xc_c, oncho_yc_c = tmp_gao.set_data_default_position(tmp_gao.I, tmp_gao.Q, tmp_gao.f)
            # tmp_phase = np.arctan2(oncho_yc_c, oncho_xc_c)
            # tmp_phase_c = self.gao_obj.phase_smoother(tmp_phase, std_theta=tmp_phase[fr_idx])


            amp_segments.append(np.column_stack([tmp_f*1E-6, tmp_log_amp]))
            phase_segments.append(np.column_stack([tmp_f*1E-6, tmp_phase_c]))
            segments.append(np.column_stack([oncho_xc_c, oncho_yc_c]))

            if(cnt==0):
                ax.scatter(oncho_xc_c[tmp_fr_idx], oncho_yc_c[tmp_fr_idx], label='fr point',color='w', edgecolor='k', zorder=2, s=8)
                ax3.scatter(tmp_f[tmp_fr_idx]*1E-6, tmp_phase_c[tmp_fr_idx], label='fr point',color='w', edgecolor='k', zorder=6, s=8)
                xlims = [[np.min(oncho_xc_c), np.max(oncho_xc_c)]]
                ylims = [[np.min(oncho_yc_c), np.max(oncho_yc_c)]]
                amp_ylims = [[np.min(tmp_log_amp), np.max(tmp_log_amp)]]
                phase_ylims= [[np.min(tmp_phase_c), np.max(tmp_phase_c)]]
                amp_phase_xlim = [np.min(tmp_f*1E-6), np.max(tmp_f*1E-6)]
            elif(cnt>0):
                ax.scatter(oncho_xc_c[tmp_fr_idx], oncho_yc_c[tmp_fr_idx], color='w', edgecolor='k', zorder=2, s=8)
                ax3.scatter(tmp_f[tmp_fr_idx]*1E-6, tmp_phase_c[tmp_fr_idx], label='fr point',color='w', edgecolor='k', zorder=6, s=8)
                xlims = np.append(xlims, [[np.min(oncho_xc_c), np.max(oncho_xc_c)]], axis=0)
                ylims = np.append(ylims, [[np.min(oncho_yc_c), np.max(oncho_yc_c)]], axis=0)
                amp_ylims = np.append(amp_ylims, [[np.min(tmp_log_amp), np.max(tmp_log_amp)]], axis=0)
                phase_ylims = np.append(phase_ylims, [[np.min(tmp_phase_c), np.max(tmp_phase_c)]], axis=0)
            #ax.plot(oncho_xc_c, oncho_yc_c, marker='o', markersize=2, zorder=1, color=cm.inferno((Tarray[cnt]-Tmin)/(Tmax-Tmin)))
            cnt += 1
        expand = 1.05
        expand_f_range = (amp_phase_xlim[1]-amp_phase_xlim[0])*0.05
        amp_max = 0.50
        expand_amp_min = 1.05
        ax.set_xlim(xlims[:, 0].min()*expand, xlims[:, 1].max()*expand)
        ax.set_ylim(ylims[:, 0].min()*expand, ylims[:, 1].max()*expand)
        ax2.set_xlim(amp_phase_xlim[0]-expand_f_range, amp_phase_xlim[1]+expand_f_range)
        ax2.set_ylim(amp_ylims[:, 0].min()*expand_amp_min, 0.50)
        ax3.set_xlim(amp_phase_xlim[0]-expand_f_range, amp_phase_xlim[1]+expand_f_range)
        ax3.set_ylim(phase_ylims[:, 0].min()*expand, phase_ylims[:, 1].max()*expand)
        norm = colors.Normalize(Tmin, Tmax)
        line_segments = LineCollection(segments, linewidths=2, cmap='inferno', norm=norm, zorder=1)
        line_segments.set_array(Tarray)
        amp_line_segments = LineCollection(amp_segments, linewidths=2, cmap='inferno', norm=norm, zorder=1)
        amp_line_segments.set_array(Tarray)
        phase_line_segments = LineCollection(phase_segments, linewidths=2, cmap='inferno', norm=norm, zorder=1)
        phase_line_segments.set_array(Tarray)
        ax.add_collection(line_segments)
        axcb = fig.colorbar(line_segments)
        axcb.set_label("Temperature (mK)")
        #self.plt_obj.gca().set_aspect('equal', adjustable='box')

        ax2.add_collection(amp_line_segments)
        axcb2 = fig2.colorbar(amp_line_segments)
        axcb2.set_label("Temperature (mK)")

        ax3.add_collection(phase_line_segments)
        axcb3 = fig3.colorbar(phase_line_segments)
        axcb3.set_label("Temperature (mK)")

        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                save_fig.savefig(self.save_dir+fig_lb+'.pdf', dpi=200)
            

    def plot_phase_shift(self, oncho_file_list, N0, delta_0, volume, **kwargs):
        options = {'save':False}
        options.update(kwargs)

        self.gao_obj.oncho_analisys(oncho_file_list, self.fr, self.Qr, **kwargs)
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
        self.tot_h = 0.0
        self.fh_skew_ther = -0.60
        self.std_ratio_ther = 0.80
        self.divide_index = 200
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
        self.failed_combined_df = pd.DataFrame([])
        self.sig_tot = 0.0
        self.nos_tot = 0.0
        self.window_length = 100
        self.data_length = 1024
        self.sig_amp = np.array([])
        self.nos_amp = np.array([])
        self.sig_ohno_array = np.array([])
        self.nos_ohno_array = np.array([])
        # for ohno area analisys
        self.sig_waste_tot = 0.0
        self.nos_waste_tot = 0.0

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
                tau, xc, yc, r, fr, Qr, phi_0 = self.gao_obj_dict[trg_file].get_fit_params()
                fit_params = [tau, xc, yc, r, fr, Qr, phi_0]
                if(self.gao_obj_dict[trg_file].lmfit_result.redchi>5):
                    fine_fit_success = False
                elif(self.gao_obj_dict[trg_file].lmfit_result.redchi<5):
                    fine_fit_success = True
            except ValueError:
                print("fine fitting : Avoid")
                fit_params = coarse_fit_params
                fine_fit_success = False
            fit_results = np.append(fit_params, fine_fit_success)
            swp_fit_params_list.append(fit_results)
            swp_fit_params_index.append(trg_file.split('/')[-2]+'/'+trg_file.split('/')[-1])
        self.swp_params_data = pd.DataFrame(swp_fit_params_list[1:], index = swp_fit_params_index, columns=swp_fit_params_header)
        print(self.swp_params_data)


    def load_trg(self, **kwargs):
        options = {
        'total_time':0.0,
        'skew_ther':-0.60, 
        'std_ratio_ther':0.80, 
        'divide_index':200,
        }
        options.update(kwargs)
        self.fh_skew_ther = options['skew_ther']
        self.std_ratio_ther = options['std_ratio_ther']
        self.divide_index = options['divide_index']
        header = ['file_name', 'succeeded', 'failed', 'total']
        fit_info_list = [header]
        for trg_file in self.trg_spr_freq_dict.keys():
            print("loading" + trg_file + "...")
            trg_set = self.gao_obj_dict[trg_file].tod2trg(trg_file, self.trg_spr_freq_dict[trg_file][0], self.trg_spr_freq_dict[trg_file][1])
            trgholder = trgfit.Trgholder(self.gao_obj_dict[trg_file].swp_file_name, trg_file, self.trg_spr_freq_dict[trg_file][0])
            trgholder.analyze_trg(trg_set, **options)
            self.trg_file_dict[trg_file] = trgholder
            tmp_row = [trg_file.split('/')[-2]+'/'+trg_file.split('/')[-1], len(trgholder.oneshot_list), len(trgholder.failed_list), len(trgholder.oneshot_list)+len(trgholder.failed_list)]
            fit_info_list.append(tmp_row)
        fit_info = pd.DataFrame(fit_info_list[1:], columns=header)
        self.fit_info = fit_info.set_index('file_name')
        self.combined_df = pd.concat([trgholder.analyzed_data for trgholder in self.trg_file_dict.values()], keys=[trg_fname.split('/')[-2]+'/'+trg_fname.split('/')[-1] for trg_fname in self.trg_file_dict.keys()])
        self.failed_combined_df = pd.concat([trgholder.analyzed_failed_data for trgholder in self.trg_file_dict.values()], keys=[trg_fname.split('/')[-2]+'/'+trg_fname.split('/')[-1] for trg_fname in self.trg_file_dict.keys()])
        if(options['total_time']==0.0):
            self.tot_h = 20*len(self.trg_file_dict.keys())/60.0
        elif(options['total_time']>0.0):
            self.tot_h = options['total_time']/60.0
        print('total_time[hour] : {0:.1f}'.format(self.tot_h))
        pd.options.display.float_format = '{:.2e}'.format

    def plot_sweep(self, **kwargs):
        options = {'save':False,
        'trg_fname':self.trg_swp_file_list[0][0],
        'loc':'upper left'}
        options.update(kwargs)

        dpi_val = 200
        plot_I = self.gao_obj_dict[options['trg_fname']].I
        plot_Q = self.gao_obj_dict[options['trg_fname']].Q
        plot_f = self.gao_obj_dict[options['trg_fname']].f
        plot_fr, plot_tau, plot_xc, plot_yc, plot_fine_fit_success = self.swp_params_data.loc[options['trg_fname'].split('/')[-2]+'/'+options['trg_fname'].split('/')[-1], ['fr', 'tau', 'xc', 'yc', 'fine_fit_success']]
        fr_index = np.argmin(np.abs(plot_fr - plot_f))

        x, y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(plot_I, plot_Q, plot_f, plot_tau)
        xc_c, yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(plot_I, plot_Q, plot_f)

        
        crs_x, crs_y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(plot_I, plot_Q, plot_f, self.gao_obj_dict[options['trg_fname']].lmfit_init_params['tau']) 
        crs_xc_c, crs_yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(crs_x, crs_y, plot_f, coarse_fit=True, xc=self.gao_obj_dict[options['trg_fname']].lmfit_init_circle_params['xc'], yc=self.gao_obj_dict[options['trg_fname']].lmfit_init_circle_params['yc'])
        crs_theta = np.arctan2(crs_yc_c, crs_xc_c)
        crs_theta_c = self.gao_obj_dict[options['trg_fname']].phase_smoother(crs_theta)

        amp_dB = 10*np.log10(np.sqrt(plot_I**2+plot_Q**2))

        phase = np.arctan2(plot_Q, plot_I)
        smooth_phase = self.gao_obj_dict[options['trg_fname']].phase_smoother(phase)

        fig_IQ = self.plt_obj.figure('IQ', figsize=(10,10))
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
        ax_IQ.scatter(xc_c[fr_index], yc_c[fr_index], label='fr', color='w', edgecolors='k', marker='^', zorder=10)

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

        self.plt_obj.legend(loc=options['loc'])
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
        self.plt_obj.legend(loc=options['loc'])

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
        trg_fig = self.plt_obj.figure('one_trg', clear=True)
        trg_ax = trg_fig.add_subplot(111)
        trg_ax.tick_params(direction='in')
        trg_ax.set_title('one trigger waveform')
        trg_ax.set_xlabel('time [$\\mu s$]')
        trg_ax.set_ylabel('phase [rad]')
        trg_ax.grid(True, zorder=0)

        plot_trgholder = self.trg_file_dict[options['trg_fname']]
        if(options['noise_plot']==True):
            if((len(plot_trgholder.failed_list)==0)|(len(plot_trgholder.failed_list)<=options['trg_index'])):
                print("Selected noise waveform could not be found.")
                print("plot first noise waveform")
                cnt = 0
                while(plot_trgholder.oneshot_list==[]):
                    options['trg_index']=0
                    options['trg_fname'] = self.trg_swp_file_list[cnt][0]
                    plot_trgholder = self.trg_file_dict[options['trg_fname']]
                    cnt += 1
                if(plot_trgholder.oneshot_list!=[]):
                    time, phase = plot_trgholder.failed_list[0].output_data()
                    trg_ax.plot(time, phase)
                else:
                    print("No noise waveform was found.")
                    print("Cannot plot noise waveform")
            elif(len(plot_trgholder.failed_list)>options['trg_index']):
                time, phase = plot_trgholder.failed_list[options['trg_index']].output_data()

        elif(options['noise_plot']==False):
            if((len(plot_trgholder.oneshot_list)==0)|(len(plot_trgholder.oneshot_list)<=options['trg_index'])):
                print("Selected trgger waveform could not be found.")
                print("plot first trigger waveform")
                cnt = 0
                while(plot_trgholder.oneshot_list==[]):
                    options['trg_index']=0
                    options['trg_fname'] = self.trg_swp_file_list[cnt][0]
                    plot_trgholder = self.trg_file_dict[options['trg_fname']]
                    cnt += 1
                if(plot_trgholder.oneshot_list!=[]):
                    fit_time_min, fit_time_max = plot_trgholder.oneshot_list[options['trg_index']].time[plot_trgholder.oneshot_list[options['trg_index']].phase_fit_range]
                    fit_time = np.linspace(fit_time_min, fit_time_max)
                    fit_phase = fn.phase_tau_func(fit_time, plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params)
                    params_row = plot_trgholder.analyzed_data.loc[options['trg_index'],:]
                    trg_ax.plot(fit_time*1e6, fit_phase, color='r', label='fit: $\\tau$ = {0:.2e} $\\mu s$'.format(plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params.valuesdict()['phase_tau']*1e6), zorder=10)
                else:
                    print("No trigger waveform was found.")
                    print("Cannot plot trigger waveform")
            elif(len(plot_trgholder.oneshot_list)>options['trg_index']):
                time, phase = plot_trgholder.oneshot_list[options['trg_index']].output_data()
                fit_time_min, fit_time_max = plot_trgholder.oneshot_list[options['trg_index']].time[plot_trgholder.oneshot_list[options['trg_index']].phase_fit_range]
                fit_time = np.linspace(fit_time_min, fit_time_max)
                fit_phase = fn.phase_tau_func(fit_time, plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params)
                params_row = plot_trgholder.analyzed_data.loc[options['trg_index'],:]
                trg_ax.plot(fit_time*1e6, fit_phase, color='r', label='fit: $\\tau$ = {0:.2e} $\\mu s$'.format(plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params.valuesdict()['phase_tau']*1e6), zorder=10)
            # else:
            #     print("Selected trgger waveform could not be found.")
            #     print("plot first trigger waveform")
            #     time, phase = plot_trgholder.oneshot_list[0].output_data()
            #     options['trg_index']=0
            #     fit_time_min, fit_time_max = plot_trgholder.oneshot_list[options['trg_index']].time[plot_trgholder.oneshot_list[options['trg_index']].phase_fit_range]
            #     fit_time = np.linspace(fit_time_min, fit_time_max)
            #     fit_phase = fn.phase_tau_func(fit_time, plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params)
            #     params_row = plot_trgholder.analyzed_data.loc[options['trg_index'],:]
            #     trg_ax.plot(fit_time*1e6, fit_phase, color='r', label='fit: $\\tau$ = {0:.2e} $\\mu s$'.format(plot_trgholder.oneshot_list[options['trg_index']].lmfit_tau_result.params.valuesdict()['phase_tau']*1e6), zorder=10)


        trg_ax.plot(time*1e6, phase, zorder=5)
        ymin, ymax = trg_ax.get_ylim()
        trg_ax.plot(np.ones(3)*time[self.divide_index]*1e6, np.linspace(ymin-1.0, ymax+1.0, 3), color='r', linestyle=':', label='divide line')
        trg_ax.set_ylim(ymin, ymax)
        self.plt_obj.legend()
        if(options['save']==True):
            save_fig = self.plt_obj.figure('one_trg')
            save_fig.savefig(self.save_dir+'one_trigger_waveform.pdf', dpi=200)

        if(options['noise_plot']==False):
            return params_row

    def check_trg(self, **kwargs):
        options = {'save':False,
        'triggered_index':0,
        'trg_fname':self.trg_swp_file_list[0][0],
        'noise_plot':False,
        'loc':'upper left'}
        options.update(kwargs)

        plot_I = self.gao_obj_dict[options['trg_fname']].I
        plot_Q = self.gao_obj_dict[options['trg_fname']].Q
        plot_f = self.gao_obj_dict[options['trg_fname']].f
        plot_fr, plot_tau, plot_xc, plot_yc, plot_fine_fit_success = self.swp_params_data.loc[options['trg_fname'].split('/')[-2]+'/'+options['trg_fname'].split('/')[-1], ['fr', 'tau', 'xc', 'yc', 'fine_fit_success']]
        fr_index = np.argmin(np.abs(plot_fr - plot_f))
        spr = self.trg_spr_freq_dict[options['trg_fname']][0]
        trg_freq = self.trg_spr_freq_dict[options['trg_fname']][1]

        x, y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(plot_I, plot_Q, plot_f, plot_tau)
        xc_c, yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(plot_I, plot_Q, plot_f)
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
        ax_IQ.scatter(xc_c[fr_index], yc_c[fr_index], label='fr', color='w', edgecolors='k', marker='^')

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

        trg_tod_data = np.genfromtxt(options['trg_fname'], delimiter=" ")

        scale_factor = spr
        trg_tod_I = trg_tod_data[:, 1]*scale_factor
        trg_tod_Q = trg_tod_data[:, 2]*scale_factor
        trg_tod_I = trg_tod_data[:, 1]*spr
        trg_tod_Q = trg_tod_data[:, 2]*spr
        xc_c, yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(trg_tod_I, trg_tod_Q, trg_freq)
        trg_header_index = np.where(trg_tod_data[:, 0]==0.0)
        if(len(trg_header_index[0])<options['triggered_index']):
            print('Selected triggered index is out of range')
            options['triggered_index'] = 0
        else:
            pass
        start = trg_header_index[0][options['triggered_index']] 
        if(options['triggered_index']==trg_header_index[0][-1]):
            stop = len(trg_tod_data[:, 0])
        else:
            stop = trg_header_index[0][options['triggered_index']+1]
        trg_I = trg_tod_I[start:stop]
        trg_Q = trg_tod_Q[start:stop]

        trg_x, trg_y = self.gao_obj_dict[options['trg_fname']].remove_tau_effect(trg_I, trg_Q, trg_freq, plot_tau)
        trg_xc_c, trg_yc_c = self.gao_obj_dict[options['trg_fname']].set_data_default_position(trg_I, trg_Q, trg_freq)
        ax_IQ.scatter(trg_I, trg_Q, color='w', edgecolors='k', marker='o')
        ax_IQ.scatter(trg_x, trg_y, color='w', edgecolors='k', marker='o')
        ax_IQ.scatter(trg_xc_c, trg_yc_c, color='w', edgecolors='k', marker='o')
        
        self.plt_obj.legend(loc=options['loc'])
        self.plt_obj.gca().set_aspect('equal', adjustable='box')



    def plot_histogram(self, **kwargs):
        # options = {'save':False,
        # 'log_yaxis':True,
        # 'tau_bins':int(round(np.log2(len(self.combined_df['phase_tau']))+1)),
        # 'tau_min':0.0,
        # 'tau_max':self.combined_df['phase_tau'].max()*1e6,
        # 'amp_bins':int(round(np.log2(len(self.combined_df['phase_Amp']))+1)),
        # 'amp_min':self.combined_df['phase_Amp'].min(),
        # 'amp_max':self.combined_df['phase_Amp'].max(),
        # 'area_bins':int(round(np.log2(len(self.combined_df['phase_area']))+1)),
        # 'area_min':0.0,
        # 'area_max':self.combined_df['phase_area'].max()*1e6,
        # 'avt_tau_min_max':[0.0, self.combined_df['phase_tau'].max()*1e6],
        # 'avt_amp_min_max':[self.combined_df['phase_Amp'].min(), self.combined_df['phase_Amp'].max()],
        # 'avt_bins':[int(round(np.log2(len(self.combined_df['phase_tau']))+1)), int(round(np.log2(len(self.combined_df['phase_Amp']))+1))],
        # 'cut':[0]}
        options = {'save':False,
        'log_yaxis':False,
        'tau_bins':100,
        'tau_min':0.0,
        'tau_max':self.combined_df['phase_tau'].max()*1e6,
        'amp_bins':100,
        'amp_min':self.combined_df['phase_Amp'].min(),
        'amp_max':self.combined_df['phase_Amp'].max(),
        'area_bins':100,
        'area_min':0.0,
        'area_max':self.combined_df['phase_area'].max()*1e6,
        'avt_tau_min_max':[0.0, self.combined_df['phase_tau'].max()*1e6],
        'avt_amp_min_max':[self.combined_df['phase_Amp'].min(), self.combined_df['phase_Amp'].max()],
        'avt_bins':[50, 50],
        'cut':[0],
        'tag_name':""}
        options.update(kwargs)
        err_kwgs = {'linestyle':':',
        'linewidth':1,
        'capsize':2,
        'zorder':5.5}

        if(len(options['cut'])==1):
            cut_df = self.combined_df
        elif(len(options['cut'])!=1):
            cut_df = self.combined_df[options['cut']]

        # num of erntries of 'phase_tau', 'phase_Amp', 'phase_area' and also 'phase_bias' are the same
        tot_weights = np.ones(len(cut_df['phase_tau']))/self.tot_h

        fig_tau = self.plt_obj.figure('tau_hist')
        ax_tau = fig_tau.add_subplot(111)
        ax_tau.set_title('$\\tau$ distoribution (events per hour)')
        ax_tau.set_xlabel('$\\tau$ [$\\mu s$]')
        ax_tau.set_ylabel('events/bin')
        ax_tau.tick_params(direction='in')
        ax_tau.grid(True, zorder=0)
        if(options['log_yaxis']==True):
            ax_tau.set_yscale('log')
        elif(options['log_yaxis']==False):
            pass
        tau_n, tau_bins, tau_patches = ax_tau.hist(cut_df['phase_tau']*1e6, bins=options['tau_bins'], range=(options['tau_min'], options['tau_max']), zorder=5, color='steelblue', weights=tot_weights)
        # tau_n, tau_bins = np.histogram(cut_df['phase_tau']*1e6, bins=options['tau_bins'], range=(options['tau_min'], options['tau_max']))
        # tau_err = np.sqrt(tau_n)
        # tau_bin_width = tau_bins[1]-tau_bins[0]
        # tau_bin_centers = (tau_bins[1:]+tau_bins[:-1])/2
        # ax_tau.bar(tau_bin_centers, tau_n/self.tot_h, width=tau_bin_width, yerr=tau_err/self.tot_h, zorder=5, color='steelblue')
        
        
        tau_n_bins = np.hstack((tau_n.reshape(-1,1), tau_bins[:-1].reshape(-1,1)))
        tau_entries = len(cut_df['phase_tau'].index)
        tau_mean = np.mean(cut_df['phase_tau'])
        tau_sig = np.std(cut_df['phase_tau'])
        rows = ['Entries', 'mean', 'std']
        tau_data = [['{0:d}/{1:.1f} hour'.format(tau_entries, self.tot_h)], ['{0:.2f} $\\mu$s'.format(tau_mean*1e6)], ['{0:.2f} $\\mu$s'.format(tau_sig*1e6)]]
        self.plt_obj.table(
            cellText=tau_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper right',
            zorder=10
        )
        # tau_bin_width = tau_bins[-1]-tau_bins[-2]
        # tau_x = tau_bins[:-1]+tau_bin_width/2
        # lognom_mod = LognormalModel()
        # lognom_pars = lognom_mod.guess(tau_n, x=tau_x)
        # lognom_pars['amplitude'].set(np.max(tau_n))
        # lognom_pars['center'].set(np.log(tau_x[np.argmax(tau_n)]))
        # lognom_out = lognom_mod.fit(tau_n, lognom_pars, x=tau_x)
        # print(lognom_out.fit_report())
        # tau_mean = np.exp(lognom_out.params.valuesdict()['center']+(lognom_out.params.valuesdict()['sigma']**2)/2)
        # fit_tau = np.linspace(tau_x[0], tau_x[-1], 200)
        # print(tau_mean)
        # ax_tau.plot(fit_tau, lognom_out.eval(x=fit_tau), zorder=6)



        fig_amp = self.plt_obj.figure('amp_hist')
        ax_amp = fig_amp.add_subplot(111)
        ax_amp.set_title('Amp. distribution (events per hour)')
        ax_amp.set_xlabel('Amp. [rad]')
        ax_amp.set_ylabel('events/bin')
        ax_amp.grid(True, zorder=0)
        if(options['log_yaxis']==True):
            ax_amp.set_yscale('log')
        elif(options['log_yaxis']==False):
            pass
        amp_n, amp_bins, amp_patches = ax_amp.hist(cut_df['phase_Amp'], bins=options['amp_bins'], range=(options['amp_min'], options['amp_max']), zorder=5, color='crimson', weights=tot_weights)
        amp_n_bins = np.hstack((amp_n.reshape(-1,1), amp_bins[:-1].reshape(-1,1)))
        amp_entries = len(cut_df['phase_Amp'].index)
        amp_mean = np.mean(cut_df['phase_Amp'])
        amp_sig = np.std(cut_df['phase_Amp'])
        rows = ['Entries', 'mean', 'std']
        amp_data = [['{0:d}/{1:.1f} hour'.format(amp_entries, self.tot_h)], ['{0:.2f} rad'.format(amp_mean)], ['{0:.2f} rad'.format(amp_sig)]]
        self.plt_obj.table(
            cellText=amp_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper right',
            zorder=10
        )

        fig_area = self.plt_obj.figure('area_hist')
        ax_area = fig_area.add_subplot(111)
        ax_area.set_title('Area distribution (events per hour)')
        ax_area.set_xlabel('area [$\\mu s \\cdot$ rad]')
        ax_area.set_ylabel('events/bin')
        ax_area.grid(True, zorder=0)
        if(options['log_yaxis']==True):
            ax_area.set_yscale('log')
        elif(options['log_yaxis']==False):
            pass
        area_n, area_bins, area_patches = ax_area.hist(cut_df['phase_area']*1e6, bins=options['area_bins'], range=(options['area_min'], options['area_max']), zorder=5, color='green', weights=tot_weights)
        area_n_bins = np.hstack((area_n.reshape(-1,1), area_bins[:-1].reshape(-1,1)))
        area_entries = len(cut_df['phase_area'].index)
        area_mean = np.mean(cut_df['phase_area'])
        area_sig = np.std(cut_df['phase_area'])
        rows = ['Entries', 'mean', 'std']
        area_data = [['{0:d}/{1:.1f} hour'.format(area_entries, self.tot_h)], ['{0:.2f} rad$\\cdot\\mu$s'.format(area_mean*1e6)], ['{0:.2f} rad$\\cdot\\mu$s'.format(area_sig*1e6)]]
        self.plt_obj.table(
            cellText=area_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper right',
            zorder=10
        )

        fig_tvsAmp = self.plt_obj.figure('tvsAmp_hist')
        ax_tvsAmp = fig_tvsAmp.add_subplot(111)
        ax_tvsAmp.set_title('$\\tau$ vs Amp histogram (events per hour)')
        ax_tvsAmp.set_xlabel('$\\tau$. [$\\mu$ s]')
        ax_tvsAmp.set_ylabel('Amp. [rad]')
        ax_tvsAmp.grid(True, zorder=0)
        tvsAmp_h, tau_tvsAmp_xedges, tau_tvsAmp_yedges, tvsAmp_im = ax_tvsAmp.hist2d(cut_df['phase_tau']*1e6, cut_df['phase_Amp'], bins=options['avt_bins'], range=[options['avt_tau_min_max'], options['avt_amp_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        tvsAmp_im.cmap.set_under('w')
        tvsAmp_cbar = self.plt_obj.colorbar(tvsAmp_im, ax=ax_tvsAmp)
        tvsAmp_cbar.set_label('events/bin')


        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                self.plt_obj.tight_layout()
                save_fig.savefig(self.save_dir+ options['tag_name'] +fig_lb+'.pdf', dpi=200, figsize=(10,10))
            np.savetxt(self.save_dir+ options['tag_name'] +'amp_hist.dat', amp_n_bins, delimiter=' ', header='n bins')
            np.savetxt(self.save_dir+ options['tag_name'] +'tau_hist.dat', tau_n_bins, delimiter=' ', header='n bins')
            np.savetxt(self.save_dir+ options['tag_name'] +'area_hist.dat', area_n_bins, delimiter=' ', header='n bins')


    def plot_stats(self, **kwargs):
        options = {
        'cut':[0]
        }
        options.update(kwargs)

        if(len(options['cut'])==1):
            cut_df = pd.concat([self.combined_df, self.failed_combined_df], axis=0, sort=False)
        elif(len(options['cut'])!=1):
            cut_df = pd.concat([self.combined_df[options['cut']], self.failed_combined_df], axis=0, sort=False)

        out_ther_entries = len(cut_df[(cut_df['fh_skew']>self.fh_skew_ther)&(cut_df['std_ratio']>self.std_ratio_ther)].index)
        out_ther_skew_mean = np.mean(cut_df[(cut_df['fh_skew']>self.fh_skew_ther)&(cut_df['std_ratio']>self.std_ratio_ther)]['fh_skew'])
        out_ther_skew_std = np.std(cut_df[(cut_df['fh_skew']>self.fh_skew_ther)&(cut_df['std_ratio']>self.std_ratio_ther)]['fh_skew'])
        out_ther_sr_mean = np.mean(cut_df[(cut_df['fh_skew']>self.fh_skew_ther)&(cut_df['std_ratio']>self.std_ratio_ther)]['std_ratio'])
        out_ther_sr_std = np.std(cut_df[(cut_df['fh_skew']>self.fh_skew_ther)&(cut_df['std_ratio']>self.std_ratio_ther)]['std_ratio'])
        in_ther_entries = len(cut_df[(cut_df['fh_skew']<=self.fh_skew_ther)|(cut_df['std_ratio']<=self.std_ratio_ther)].index)
        in_ther_skew_mean = np.mean(cut_df[(cut_df['fh_skew']<=self.fh_skew_ther)|(cut_df['std_ratio']<=self.std_ratio_ther)]['fh_skew'])
        in_ther_skew_std = np.std(cut_df[(cut_df['fh_skew']<=self.fh_skew_ther)|(cut_df['std_ratio']<=self.std_ratio_ther)]['fh_skew'])
        in_ther_sr_mean = np.mean(cut_df[(cut_df['fh_skew']<=self.fh_skew_ther)|(cut_df['std_ratio']<=self.std_ratio_ther)]['std_ratio'])
        in_ther_sr_std = np.std(cut_df[(cut_df['fh_skew']<=self.fh_skew_ther)|(cut_df['std_ratio']<=self.std_ratio_ther)]['std_ratio'])
        
        columns = ['skewness(inner)', 'std ratio(inner)', 'skewness(outer)', 'std ratio(outer)']
        rows = ['mean', 'std']
        stats_data = [['{0:.2f}'.format(in_ther_skew_mean), '{0:.3f}'.format(in_ther_sr_mean), '{0:.3f}'.format(out_ther_skew_mean), '{0:.3f}'.format(out_ther_sr_mean)], ['{0:.3f}'.format(in_ther_skew_std), '{0:.3f}'.format(in_ther_sr_std), '{0:.3f}'.format(out_ther_skew_std), '{0:.3f}'.format(out_ther_sr_std)]]

        
        options2 = {
        'save':False,
        'svsr_sr_min_max':[cut_df['std_ratio'].min(), cut_df['std_ratio'].max()],
        'svsr_skew_min_max':[cut_df['fh_skew'].min(), cut_df['fh_skew'].max()],
        'svsr_bins':[int(round(np.log2(len(cut_df['fh_skew']))+1)), int(round(np.log2(len(cut_df['std_ratio']))+1))],
        'svt_tau_min_max':[0.0, cut_df['phase_tau'].max()*1e6],
        'svt_skew_min_max':[cut_df['fh_skew'].min(), cut_df['fh_skew'].max()],
        'svt_bins':[int(round(np.log2(len(cut_df['fh_skew']))+1)), int(round(np.log2(len(cut_df['phase_tau']))+1))],
        'srvt_tau_min_max':[0.0, cut_df['phase_tau'].max()*1e6],
        'srvt_sr_min_max':[cut_df['std_ratio'].min(), cut_df['std_ratio'].max()],
        'srvt_bins':[int(round(np.log2(len(cut_df['std_ratio']))+1)), int(round(np.log2(len(cut_df['phase_tau']))+1))],
        'svA_Amp_min_max':[cut_df['phase_Amp'].min(), cut_df['phase_Amp'].max()],
        'svA_skew_min_max':[cut_df['fh_skew'].min(), cut_df['fh_skew'].max()],
        'svA_bins':[int(round(np.log2(len(cut_df['fh_skew']))+1)), int(round(np.log2(len(cut_df['phase_Amp']))+1))],
        'srvA_Amp_min_max':[cut_df['phase_Amp'].min(), cut_df['phase_Amp'].max()],
        'srvA_sr_min_max':[cut_df['std_ratio'].min(), cut_df['std_ratio'].max()],
        'srvA_bins':[int(round(np.log2(len(cut_df['std_ratio']))+1)), int(round(np.log2(len(cut_df['phase_Amp']))+1))],
        'tag_name':""
        }
        options.update(options2)
        options.update(kwargs)

        tot_weights = np.ones(len(cut_df['fh_skew']))/self.tot_h

        fig_svsr = self.plt_obj.figure('svsr_hist')
        ax_svsr = fig_svsr.add_subplot(111)
        ax_svsr.set_title('first half skew vs std_ratio histogram (events per hour)')
        ax_svsr.set_ylabel('std ratio [no unit]')
        ax_svsr.set_xlabel('first half skew [no unit]')
        ax_svsr.grid(True, zorder=0)
        svsr_h, svsr_xedges, svsr_yedges, svsr_im = ax_svsr.hist2d(cut_df['fh_skew'], cut_df['std_ratio'], bins=options['svsr_bins'], range=[options['svsr_skew_min_max'], options['svsr_sr_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        svsr_cbar = self.plt_obj.colorbar(svsr_im, ax=ax_svsr)
        svsr_cbar.set_label('events/bin')
        svsr_ymin, svsr_ymax = ax_svsr.get_ylim()
        svsr_xmin, svsr_xmax = ax_svsr.get_xlim()
        ax_svsr.plot(self.fh_skew_ther*np.ones(3), np.linspace(svsr_ymin-1.0, self.std_ratio_ther, 3) ,color='r', linestyle=':', zorder=10, label='threshold')
        ax_svsr.plot(np.linspace(svsr_xmin-1.0, self.fh_skew_ther, 3), self.std_ratio_ther*np.ones(3) ,color='r', linestyle=':', zorder=10)
        ax_svsr.set_ylim(svsr_ymin, svsr_ymax)
        ax_svsr.set_xlim(svsr_xmin, svsr_xmax)
        self.plt_obj.table(
            cellText=stats_data,
            cellLoc='center',
            colWidths=[0.3, 0.3, 0.3, 0.3],
            colLabels=columns,
            rowLabels=rows,
            rowLoc='center',
            loc='bottom', 
            bbox=[0.0,-0.35, 1.0, 0.18],
            zorder=10
        )
        self.plt_obj.tight_layout()
        # self.plt_obj.table(
        #     cellText=in_data,
        #     cellLoc='center',
        #     colWidths=[0.3, 0.3],
        #     colLabels=columns,
        #     rowLabels=rows,
        #     rowLoc='center',
        #     loc='lower right',
        #     zorder=10
        # )
        ax_svsr.legend(loc='upper left')

        fig_svt = self.plt_obj.figure('svt_hist')
        ax_svt = fig_svt.add_subplot(111)
        ax_svt.set_title('first half skew vs $\\tau$ histogram (events per hour)')
        ax_svt.set_ylabel('$\\tau$ [$\\mu$ s]')
        ax_svt.set_xlabel('first half skew. [no unit]')
        ax_svt.grid(True, zorder=0)
        svt_h, svt_xedges, svt_yedges, svt_im = ax_svt.hist2d(cut_df['fh_skew'], cut_df['phase_tau']*1e6, bins=options['svt_bins'], range=[options['svt_skew_min_max'], options['svt_tau_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        svt_cbar = self.plt_obj.colorbar(svt_im, ax=ax_svt)
        svt_cbar.set_label('events/bin')
        svt_ymin, svt_ymax = ax_svt.get_ylim()
        ax_svt.plot(self.fh_skew_ther*np.ones(3), np.linspace(svt_ymin-1.0, svt_ymax+1.0, 3), color='r', linestyle=':', label='skew threshold', zorder=6)
        ax_svt.set_ylim(svt_ymin, svt_ymax)
        ax_svt.legend()

        fig_srvt = self.plt_obj.figure('srvt_hist')
        ax_srvt = fig_srvt.add_subplot(111)
        ax_srvt.set_title('std ratio vs $\\tau$ histogram (events per hour)')
        ax_srvt.set_ylabel('$\\tau$ [$\\mu$ s]')
        ax_srvt.set_xlabel('std ratio [no unit]')
        ax_srvt.grid(True, zorder=0)
        srvt_h, srvt_xedges, srvt_yedges, srvt_im = ax_srvt.hist2d(cut_df['std_ratio'], cut_df['phase_tau']*1e6, bins=options['srvt_bins'], range=[options['srvt_sr_min_max'], options['srvt_tau_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        srvt_cbar = self.plt_obj.colorbar(srvt_im, ax=ax_srvt)
        srvt_cbar.set_label('events/bin')
        srvt_ymin, srvt_ymax = ax_srvt.get_ylim()
        ax_srvt.plot(self.std_ratio_ther*np.ones(3), np.linspace(srvt_ymin-1.0, srvt_ymax+1.0, 3), color='r', linestyle=':', label='std ratio threshold', zorder=6)
        ax_srvt.set_ylim(srvt_ymin, srvt_ymax)
        ax_srvt.legend()

        fig_svA = self.plt_obj.figure('svA_hist')
        ax_svA = fig_svA.add_subplot(111)
        ax_svA.set_title('first half skew vs Amp. histogram (events per hour)')
        ax_svA.set_ylabel('Amp. [rad]')
        ax_svA.set_xlabel('first half skew. [no unit]')
        ax_svA.grid(True, zorder=0)
        svA_h, svA_xedges, svA_yedges, svA_im = ax_svA.hist2d(cut_df['fh_skew'], cut_df['phase_Amp'], bins=options['svA_bins'], range=[options['svA_skew_min_max'], options['svA_Amp_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        svA_cbar = self.plt_obj.colorbar(svA_im, ax=ax_svA)
        svA_cbar.set_label('events/bin')
        svA_ymin, svA_ymax = ax_svA.get_ylim()
        ax_svA.plot(self.fh_skew_ther*np.ones(3), np.linspace(svA_ymin-1.0, svA_ymax+1.0, 3), color='r', linestyle=':', label='skew threshold', zorder=6)
        ax_svA.set_ylim(svA_ymin, svA_ymax)
        ax_svA.legend()

        fig_srvA = self.plt_obj.figure('srvA_hist')
        ax_srvA = fig_srvA.add_subplot(111)
        ax_srvA.set_title('std ratio vs Amp. histogram (events per hour)')
        ax_srvA.set_ylabel('Amp. [rad]')
        ax_srvA.set_xlabel('std ratio [no unit]')
        ax_srvA.grid(True, zorder=0)
        srvA_h, srvA_xedges, srvA_yedges, srvA_im = ax_srvA.hist2d(cut_df['std_ratio'], cut_df['phase_Amp'], bins=options['srvA_bins'], range=[options['srvA_sr_min_max'], options['srvA_Amp_min_max']], cmap='jet', zorder=5, weights=tot_weights, cmin=1.0E-7)
        srvA_cbar = self.plt_obj.colorbar(srvA_im, ax=ax_srvA)
        srvA_cbar.set_label('events/bin')
        srvA_ymin, srvA_ymax = ax_srvA.get_ylim()
        ax_srvA.plot(self.std_ratio_ther*np.ones(3), np.linspace(srvA_ymin-1.0, srvA_ymax+1.0, 3), color='r', linestyle=':', label='std ratio threshold', zorder=6)
        ax_srvA.set_ylim(srvA_ymin, srvA_ymax)
        ax_srvA.legend()

        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                self.plt_obj.tight_layout()
                save_fig.savefig(self.save_dir+options['tag_name'] +fig_lb+'.pdf', dpi=200)

    def tod_analyze(self, **kwargs):
        header_length=100
        a1_split_indices = np.arange(0, self.data_length, self.window_length)[1:]
        a2_split_indices = np.arange(int(self.window_length/2), self.data_length, self.window_length)[1:]
        for trgholder in self.trg_file_dict.values():
            dt = 1.0/(trgholder.sample_rate*1e3)
            self.sig_tot += len(trgholder.oneshot_list)*self.data_length*dt
            self.nos_tot += len(trgholder.failed_list)*self.data_length*dt
            if(len(trgholder.oneshot_list)!=0):
                for signal in trgholder.oneshot_list:
                    tmp_base = np.mean(signal.phase[:header_length])
                    self.sig_amp = np.append(self.sig_amp, signal.phase-tmp_base)
                    sig_a1_range = np.split(signal.phase-tmp_base, a1_split_indices)
                    sig_a2_range = np.split(signal.phase-tmp_base, a2_split_indices)
                    self.sig_ohno_array = np.append(self.sig_ohno_array, np.array([np.sum(seg) for seg in sig_a1_range[2:-1]])*dt)
                    self.sig_ohno_array = np.append(self.sig_ohno_array, np.array([np.sum(seg) for seg in sig_a2_range[2:-1]])*dt)
                    self.sig_waste_tot += (len(sig_a1_range[1]) + len(sig_a1_range[-1]))*dt
            if(len(trgholder.failed_list)!=0):
                for noise in trgholder.failed_list:
                    tmp_base = np.mean(noise.phase[:header_length])
                    self.nos_amp = np.append(self.nos_amp, noise.phase-tmp_base)
                    nos_a1_range = np.split(noise.phase-tmp_base, a1_split_indices)
                    nos_a2_range = np.split(noise.phase-tmp_base, a2_split_indices)
                    self.nos_ohno_array = np.append(self.nos_ohno_array, np.array([np.sum(seg) for seg in nos_a1_range[2:-1]])*dt)
                    self.nos_ohno_array = np.append(self.nos_ohno_array, np.array([np.sum(seg) for seg in nos_a2_range[2:-1]])*dt)
                    self.nos_waste_tot += (len(nos_a1_range[1])+len(nos_a1_range[-1]))*dt

    def plot_tod_histogram(self, **kwargs):
        options = {'save':False,
        'sn_amp_bins':int(round(np.log2(len(self.sig_amp)+len(self.nos_amp))+1)),
        'sn_amp_min':self.sig_amp.min(),
        'sn_amp_max':self.sig_amp.max(),
        'ohno_area_bins':int(round(np.log2(len(self.sig_ohno_array)+len(self.nos_ohno_array))+1)),
        'ohno_area_min':self.sig_ohno_array.min()*1e6,
        'ohno_area_max':self.nos_ohno_array.max()*1e6,
        'cut':[0],
        'tag_name':""}
        options.update(kwargs)

        sig_amp_w = np.ones(len(self.sig_amp))/self.sig_tot
        nos_amp_w = np.ones(len(self.nos_amp))/self.nos_tot
        sig_ohno_w = 100*np.ones(len(self.sig_ohno_array))/(self.sig_tot-self.sig_waste_tot)
        nos_ohno_w = 100*np.ones(len(self.nos_ohno_array))/(self.nos_tot-self.nos_waste_tot)
        
        fig_sn_amp = self.plt_obj.figure('sn_amp_hist')
        ax_sn_amp = fig_sn_amp.add_subplot(111)
        ax_sn_amp.set_title('signal and noise phase shift histgram (base subtructed)')
        ax_sn_amp.set_xlabel('Phase [rad]')
        ax_sn_amp.set_ylabel('events/bin/s')
        ax_sn_amp.set_yscale('log')
        ax_sn_amp.grid(True, zorder=0)
        sig_amp_n, sig_amp_bins, sig_amp_patches = ax_sn_amp.hist(self.sig_amp, bins=options['sn_amp_bins'], range=(options['sn_amp_min'], options['sn_amp_max']), zorder=5, label='signal', weights=sig_amp_w, color='steelblue', histtype='step')
        nos_amp_n, nos_amp_bins, nos_amp_patches = ax_sn_amp.hist(self.nos_amp, bins=options['sn_amp_bins'], range=(options['sn_amp_min'], options['sn_amp_max']), zorder=6, label='noise', weights=nos_amp_w, color='orange', histtype='step')
        sig_amp_n_bins = np.hstack((sig_amp_n.reshape(-1,1), sig_amp_bins[:-1].reshape(-1,1)))
        nos_amp_n_bins = np.hstack((nos_amp_n.reshape(-1,1), nos_amp_bins[:-1].reshape(-1,1)))
        amp_n_bins = np.hstack((sig_amp_n_bins, nos_amp_n_bins))
        sig_amp_entries = len(self.sig_amp)
        nos_amp_entries = len(self.nos_amp)
        sig_amp_mean = np.mean(self.sig_amp)
        nos_amp_mean = np.mean(self.nos_amp)
        sig_amp_sig = np.std(self.sig_amp)
        nos_amp_sig = np.std(self.nos_amp)
        columns = ['signal', 'noise']
        rows = ['Entries', 'mean', 'std']
        sn_amp_data = [['{0:d}/{1:.2f} s'.format(sig_amp_entries, self.sig_tot), '{0:d}/{1:.2f} s'.format(nos_amp_entries, self.nos_tot)], ['{0:.2e} rad'.format(sig_amp_mean), '{0:.2e} rad'.format(nos_amp_mean)], ['{0:.2e} rad'.format(sig_amp_sig), '{0:.2e} rad'.format(nos_amp_sig)]]
        self.plt_obj.table(
            cellText=sn_amp_data,
            cellLoc='center',
            colWidths=[0.3, 0.3],
            colLabels=columns,
            rowLabels=rows,
            rowLoc='center',
            loc='bottom', 
            bbox=[0.10,-0.35, 0.80, 0.18],
            zorder=10
        )
        self.plt_obj.legend(loc='upper left')

        fig_sum_sig_area = self.plt_obj.figure('sum_area_sig_hist')
        ax_sum_sig_area = fig_sum_sig_area.add_subplot(111)
        ax_sum_sig_area.set_title('ohno\'s area (sum window length: {0:d}$\\mu$s) histogram : signal'.format(self.window_length))
        ax_sum_sig_area.set_xlabel('area [rad $\\cdot \\mu$s]')
        ax_sum_sig_area.set_ylabel('events/bin/100s')
        ax_sum_sig_area.set_yscale('log')
        ax_sum_sig_area.grid(True, zorder=0)
        ohno_sig_n, ohno_sig_bins, ohno_sig_patches = ax_sum_sig_area.hist(self.sig_ohno_array*1e6, bins=options['ohno_area_bins'], range=(options['ohno_area_min'], options['ohno_area_max']), zorder=5, weights=sig_ohno_w,  color='steelblue', label='signal')
        sig_ohno_entries = len(self.sig_ohno_array)
        sig_ohno_mean = np.mean(self.sig_ohno_array)
        sig_ohno_sig = np.std(self.sig_ohno_array)
        rows = ['Entries', 'mean', 'std']
        sig_ohno_data = [['{0:d}/{1:.2f} s'.format(sig_ohno_entries, self.sig_tot-self.sig_waste_tot)], ['{0:.2e} rad$\\cdot\\mu s$'.format(sig_ohno_mean*1e6)], ['{0:.2e} rad$\\cdot\\mu s$'.format(sig_ohno_sig*1e6)]]
        self.plt_obj.table(
            cellText=sig_ohno_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper left',
            bbox=[0.15, 0.80, 0.32, 0.18],
            zorder=10
        )
        #self.plt_obj.legend()

        fig_sum_nos_area = self.plt_obj.figure('sum_area_nos_hist')
        ax_sum_nos_area = fig_sum_nos_area.add_subplot(111)
        ax_sum_nos_area.set_title('ohno\'s area (sum window length: {0:d}$\\mu$s) histogram : noise'.format(self.window_length))
        ax_sum_nos_area.set_xlabel('area [rad $\\cdot \\mu$s]')
        ax_sum_nos_area.set_ylabel('events/bin/100s')
        ax_sum_nos_area.set_yscale('log')
        ax_sum_nos_area.grid(True, zorder=0)
        ohno_nos_n, ohno_nos_bins, ohno_nos_patches = ax_sum_nos_area.hist(self.nos_ohno_array*1e6, bins=options['ohno_area_bins'], range=(options['ohno_area_min'], options['ohno_area_max']), zorder=5, weights=nos_ohno_w,  color='orange', label='noise')
        nos_ohno_entries = len(self.nos_ohno_array)
        nos_ohno_mean = np.mean(self.nos_ohno_array)
        nos_ohno_sig = np.std(self.nos_ohno_array)
        rows = ['Entries', 'mean', 'std']
        nos_ohno_data = [['{0:d}/{1:.2f} s'.format(nos_ohno_entries, self.nos_tot-self.nos_waste_tot)], ['{0:.2e} rad$\\cdot\\mu s$'.format(nos_ohno_mean*1e6)], ['{0:.2e} rad$\\cdot\\mu s$'.format(nos_ohno_sig*1e6)]]
        self.plt_obj.table(
            cellText=nos_ohno_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper left',
            bbox=[0.15, 0.80, 0.32, 0.18],
            zorder=10
        )
        #self.plt_obj.legend()

        comb_ohno = np.append(self.sig_ohno_array, self.nos_ohno_array)
        comb_tot = self.sig_tot+self.nos_tot-self.nos_waste_tot-self.sig_waste_tot
        comb_ohno_w = 100*np.ones(len(comb_ohno))/comb_tot
        fig_sum_area = self.plt_obj.figure('sum_area_hist')
        ax_sum_area = fig_sum_area.add_subplot(111)
        ax_sum_area.set_title('ohno\'s area (sum window length: {0:d}$\\mu$s) histogram : signal + noise'.format(self.window_length))
        ax_sum_area.set_xlabel('area [rad $\\cdot \\mu$s]')
        ax_sum_area.set_ylabel('events/bin/100s')
        ax_sum_area.set_yscale('log')
        ax_sum_area.grid(True, zorder=0)
        ohno_n, ohno_bins, ohno_patches = ax_sum_area.hist(comb_ohno*1e6, bins=options['ohno_area_bins'], range=(options['ohno_area_min'], options['ohno_area_max']), zorder=5, weights=comb_ohno_w,  color='green', label='combined')
        comb_ohno_entries = len(comb_ohno)
        comb_ohno_mean = np.mean(comb_ohno)
        comb_ohno_sig = np.std(comb_ohno)
        rows = ['Entries', 'mean', 'std']
        comb_ohno_data = [['{0:d}/{1:.2f} s'.format(comb_ohno_entries, comb_tot)], ['{0:.2e} rad$\\cdot\\mu s$'.format(comb_ohno_mean*1e6)], ['{0:.2e} rad$\\cdot\\mu s$'.format(comb_ohno_sig*1e6)]]
        self.plt_obj.table(
            cellText=comb_ohno_data,
            cellLoc='center',
            colWidths=[0.3],
            rowLabels=rows,
            rowLoc='center',
            loc='upper left',
            bbox=[0.15, 0.80, 0.32, 0.18],
            zorder=10
        )
        #self.plt_obj.legend(loc='upper right')
        
        if(options['save']==True):
            for fig_lb in self.plt_obj.get_figlabels():
                save_fig = self.plt_obj.figure(fig_lb)
                self.plt_obj.tight_layout()
                save_fig.savefig(self.save_dir+fig_lb+'.pdf', dpi=200)
            np.savetxt(self.save_dir +options['tag_name'] + 'tod_amp_hist.dat', amp_n_bins, delimiter=' ', header='signal_n signal_bins noise_n noise_bins')

