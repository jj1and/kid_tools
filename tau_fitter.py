import numpy as np
from . import gaopy
from . import functions as fn
from lmfit import minimizer, Parameters, report_fit

# class FitContain(object):
#     def __init__(self, swp_file_name, fit_file_name, start_idx, stop_idx, lmfit_result):
#         self.swp_file_name = swp_file_name
#         self.fit_file_name = fit_file_name
#         self.start_idx = start_idx
#         self.stop_idx = stop_idx
#         self.lmfit_result = lmfit_result


class OneTrg():
    def __init__(self, swp_file_name, trg_file_name, trg_freq, sample_rate):       
        