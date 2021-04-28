import paf_config as paf_cfg
import paf_functions as paf_fcn
import numpy as np
from scipy.signal import find_peaks, peak_widths
import pandas as pd
import fnmatch
import os
import matplotlib.pyplot as plt
from mne.io import read_raw_fif
from mne import Report


def run_paf(psd, freqs, filter_width, filter_order, diffp): # main body that interfaces with paf_functions

    freq_res = np.diff(freqs).mean() # frequency resolution

    subject_peak_freqs = []
    subject_peak_qweights = []
    smooth_psd = np.zeros(psd.shape)
    # loop through each channel and analyze/filter/model it
    for channel_idx in range(psd.shape[0]):  # loop through each channel, and perform the analysis

        signal_filtered, pmin = paf_fcn.filter_signal_and_calculate_pmin(freqs, psd[channel_idx, :], filter_width, filter_order)
        smooth_psd[channel_idx, :] = signal_filtered
        # peak-finding...
        peak_inds, _ = find_peaks(signal_filtered, height=pmin) # only find peaks greater than the model has predicted
        if peak_inds.size == 0:
            continue # no peaks detected... skip this channel and move on to the next one
        # return minima characteristics for each peak
        peak_bounds = peak_widths(signal_filtered, peak_inds, rel_height=0.5)

        channel_peak_freqs, channel_peak_qweights = paf_fcn.find_channel_peak_freq_and_qweight(freqs, signal_filtered,
                                                                                               peak_inds, peak_bounds, freq_res, diffp)
        # check if any peaks in the channel were viable enough to be considered PAF...
        if channel_peak_qweights.size == 0:
            channel_peak_freq = 0
            channel_peak_qweight = 0
        else:
            # determine the most significant peak in the channel
            channel_peak_freq = channel_peak_freqs[np.argmax(channel_peak_qweights)]
            channel_peak_qweight = channel_peak_qweights[np.argmax(channel_peak_qweights)]

        # append these to the subject-wide list...
        subject_peak_freqs.append(channel_peak_freq)
        subject_peak_qweights.append(channel_peak_qweight)

    subject_peak_freqs = np.asarray(subject_peak_freqs)
    subject_peak_qweights = np.asarray(subject_peak_qweights)

    return smooth_psd, subject_peak_freqs, subject_peak_qweights


if __name__ == '__main__':

    rep = Report() # use MNE reports to test cases, view data
    dataframe = pd.read_csv(paf_cfg.csv_loc)

    for _, subj_row in dataframe.iterrows():
        subj = subj_row['subject_id']
        age = subj_row['age_visit_1']
        diagnosis = 'TD' if subj_row['diagnosis'] == 0 else 'ASD'

        raw_match = fnmatch.filter(os.listdir(), f'{subj}*raw_sss.fif')
        if len(raw_match) == 0:
            continue
        raw_data = read_raw_fif(raw_match[0])

        psd, freqs = paf_fcn.compute_psd(raw_data, paf_cfg.occipital_subselection_modern, paf_cfg.occipital_subselection_legacy,
                                         paf_cfg.alpha_band, paf_cfg.n_fft, paf_cfg.desired_freq_res, paf_cfg.n_jobs)

        psd_smooth, peak_freqs, peak_qweights = run_paf(psd, freqs, paf_cfg.filter_width, paf_cfg.filter_order, paf_cfg.diffp)

        channel_weights, ranked_peak_freqs = paf_fcn.determine_channel_weights_and_rank_peaks_by_qweight(peak_qweights, peak_freqs)

        fig = paf_fcn.make_figure_for_report(freqs, psd, psd_smooth, ranked_peak_freqs, paf_cfg.rank_by_colors, subj, age)
        rep.add_figs_to_section(fig, captions=subj, section=diagnosis)
        plt.close(fig)

    rep.save(paf_cfg.rep_name)






