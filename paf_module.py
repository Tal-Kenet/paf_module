import paf_config as paf_cfg
import mne
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
import pandas as pd
import fnmatch
import os
import matplotlib.pyplot as plt

def rank_peaks_by_qweight(channel_weights, peak_freqs):
    """
    rank peaks in descending order (high to low) by their Q-weight (max Q-weight normalized)
    :param channel_weights: normalized Q-weights for each channel's peak alpha frequency
    :param peak_freqs: peak frequencies for each channel
    :return: unique, sorted (descending order) peak frequencies for a subject's channels
    """
    sorted_channel_weights = np.argsort(channel_weights)[::-1]
    sorted_peak_freqs = peak_freqs[sorted_channel_weights] # peak frequencies
    _, uniques_idx = np.unique(sorted_peak_freqs, return_index=True)
    sorted_unique_peak_freqs = sorted_peak_freqs[np.sort(uniques_idx)]

    return sorted_unique_peak_freqs



def gather_peak_metrics(iter, freqs, signal, peak_inds, peak_bounds, freq_res):
    """
    determine frequency span of each peak, as well as integrating the region bound by the peak's extrema
    :param iter: current index value to access location of peak
    :param freqs: frequency array
    :param signal: filtered channel PSD array
    :param peak_inds: indices of peaks
    :param peak_bounds: nested arrays containing: minimia widths, minima heights, left and right interpolated indices
    :param freq_res: frequency resolution
    :return: the peak frequency, and its Q-weight
    """

    peak_idx = peak_inds[iter]

    peak_height = signal[peak_idx]
    peak_freq = freqs[peak_idx]

    minima_width = peak_bounds[0][iter]
    minima_height = peak_bounds[1][iter]
    left_minima_sample = peak_bounds[2][iter]
    right_minima_sample = peak_bounds[3][iter]

    # interpolate frequency value for each minima
    left_minima_freq = freqs[peak_idx] - (freq_res * (peak_idx - left_minima_sample))
    right_minima_freq = freqs[peak_idx] + np.abs(freq_res * (peak_idx - right_minima_sample))

    # aggregate the extrema
    extrema_freqs = np.array([left_minima_freq, peak_freq, right_minima_freq])
    extrema_heights = np.array([minima_height, peak_height, minima_height])

    # final metrics...
    au_peak = np.trapz(extrema_heights, x=extrema_freqs)  # find the area under the peak (AUP)...
    peak_q = au_peak / (freq_res * minima_width)  # scale by the frequency span of the peak

    return peak_freq, peak_q


def find_channel_peak_freq_and_qweight(freqs, signal, peak_inds, peak_bounds, freq_res):
    """
    wrapper function for gather_peak_metrics() above
    :param freqs: frequency array
    :param signal: filtered channel PSD array
    :param peak_inds: indices of peaks
    :param peak_bounds: nested arrays containing: minimia widths, minima heights, left and right interpolated indices
    :param freq_res: frequency resolution
    :return: peak frequencies and their corresponding Q-weights for an entire channel
    """
    channel_peak_freqs = []
    channel_peak_qweights = []

    # collect data on each peak...
    for i in range(peak_inds.size):
        peak_freq, peak_q = gather_peak_metrics(i, freqs, signal, peak_inds, peak_bounds, freq_res)

        channel_peak_freqs.append(peak_freq)
        channel_peak_qweights.append(peak_q)

    return channel_peak_freqs, channel_peak_qweights


def filter_signal_and_calculate_pmin(freqs, signal, window_len, filter_order):
    """
    clean up the channel PSD and determine the minimum power necessary to identify a peak
    :param freqs: frequency array
    :param signal: channel PSD array
    :param window_len: filter frame width (in samples)
    :param filter_order: filter polynomial degree
    :return: filtered channel PSD array, minimum power value to qualify maxima as peak candidates
    """
    signal /= signal.mean() # mean-normalize channel PSD

    # fit regression line, then predict power values
    fit = np.polyfit(freqs, np.log(signal), deg=1)
    pred = np.polyval(fit, freqs) # determine pmin using these predicted values

    # minimum power value that a local maximum must exceed to qualify as a candidate
    # pmin may return as NaN/0... must implement handling of this case
    pmin = pred.std() + pred.mean()  # defined as some number X standard deviatons above the power estimate predicted by the model

    signal_filtered = savgol_filter(signal, window_length=window_len, polyorder=filter_order) # apply SGF to PSD
    return signal_filtered, pmin


def compute_psd(raw, picks_modern, picks_legacy, alpha_window, fft_length, desired_freq_res, n_jobs):
    """
    :param evoked: mne.Evoked object
    :param picks: selection of channels to parse
    :param alpha_window: lower and upper bounds to frequency range spanning alpha oscillations
    :param cropped: if True, crop the frequency and PSD array returned to the range of alpha_window (DEFAULT: TRUE)
    :return: PSD array (n_channels, n_freqs), frequency array (n_freqs,)
    """
    if raw.info['sfreq'] / fft_length > desired_freq_res:
        raw.resample(desired_freq_res * fft_length, n_jobs=n_jobs, verbose=False)
    try:
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=alpha_window[0], fmax=alpha_window[1], n_fft=fft_length,
                                                  n_jobs=n_jobs, picks=picks_modern)
    except ValueError:
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=alpha_window[0], fmax=alpha_window[1], n_fft=fft_length,
                                                  n_jobs=n_jobs, picks=picks_legacy)
    del raw
    return psd, freqs


def run_paf(raw): # main body that interacts with sensor_space_analysis module

    psd, freqs = compute_psd(raw, paf_cfg.occipital_subselection_modern, paf_cfg.occipital_subselection_legacy,
                             alpha_window=paf_cfg.alpha_band, fft_length=paf_cfg.n_fft, desired_freq_res=paf_cfg.desired_freq_res, n_jobs=8)

    freq_res = np.diff(freqs).mean() # frequency resolution

    subject_peak_freqs = []
    subject_peak_qweights = []
    smooth_psd = np.zeros(psd.shape)
    # loop through each channel and analyze/filter/model it
    for channel_idx in range(psd.shape[0]):  # loop through each channel, and perform the analysis

        signal_filtered, pmin = filter_signal_and_calculate_pmin(freqs, psd[channel_idx, :], paf_cfg.sgf_width, paf_cfg.k)
        smooth_psd[channel_idx, :] = signal_filtered
        # peak-finding...
        peak_inds, _ = find_peaks(signal_filtered, height=pmin) # only find peaks greater than the model has predicted
        if peak_inds.size == 0:
            continue # no peaks detected... skip this channel and move on to the next one
        # return minima characteristics for each peak
        peak_bounds = peak_widths(signal_filtered, peak_inds, rel_height=0.5)

        channel_peak_freqs, channel_peak_qweights = find_channel_peak_freq_and_qweight(freqs, signal_filtered,
                                                                                       peak_inds, peak_bounds, freq_res)

        # determine the most significant peak in the channel
        channel_peak_freq = channel_peak_freqs[np.argmax(channel_peak_qweights)]
        channel_peak_qweight = channel_peak_qweights[np.argmax(channel_peak_qweights)]

        # append these to the subject-wide list...
        subject_peak_freqs.append(channel_peak_freq)
        subject_peak_qweights.append(channel_peak_qweight)

    subject_peak_freqs = np.asarray(subject_peak_freqs)
    subject_peak_qweights = np.asarray(subject_peak_qweights)
    # perform cross-channel averaging
    channel_weights = subject_peak_qweights / subject_peak_qweights.max()
    # not so sure about the below... it seems very passive and average-y...
    #paf = np.sum(subject_peak_freqs * channel_weights) / np.sum(channel_weights) # MATLAB formula...?
    
    ranked_peaks = rank_peaks_by_qweight(channel_weights, subject_peak_freqs) # max Q in first index

    return freqs, psd, smooth_psd, subject_peak_freqs, channel_weights, ranked_peaks


if __name__ == '__main__':

    rep = mne.Report() # use MNE reports to test cases
    rep_name = f'PAF_module_case_testing_rawsignal_filterlength{paf_cfg.sgf_width}_polyorder{paf_cfg.k}.html'

    df = pd.read_csv('prelim_fixation_age_vs_peak.csv')
    for _, subj_row in df.iterrows():
        subj = subj_row['subject_id']
        age = subj_row['age_visit_1']
        diagnosis = 'TD' if subj_row['diagnosis'] == 0 else 'ASD'

        if len(subj) == 4:
            subject = '00' + subj
        elif len(subj) == 5:
            subject = '0' + subj
        else:
            subject = subj

        raw_match = fnmatch.filter(os.listdir(), f'{subject}*raw_sss.fif')
        if len(raw_match) == 0:
            continue
        subject_raw = mne.io.read_raw_fif(raw_match[0])

        freqs, psd, smooth_psd, subject_peak_freqs, channel_weights, ranked_peaks = run_paf(subject_raw)

        top_n_ranked_peaks = ranked_peaks[:paf_cfg.n_peaks_to_rank]
        fig, (ax_orig, ax_smooth) = plt.subplots(2, 1, sharex=True, sharey=True)
        for channel_orig in psd:
            ax_orig.plot(freqs, channel_orig)
        ax_orig.set_ylabel('PSD (original)')
        ax_orig.set_ylim((0, psd.max() + 1.5))
        
        for channel_smooth in smooth_psd:
            ax_smooth.plot(freqs, channel_smooth)
        ax_smooth.set_xlabel('Frequency [Hz]')
        ax_smooth.set_ylabel('Smoothed PSD')
        ax_smooth.set_ylim((0, psd.max() + 1.5))

        for peak_rank, peak_frequency in enumerate(top_n_ranked_peaks):
            ax_orig.axvline(peak_frequency, ls='--', c=paf_cfg.rank_by_colors[peak_rank],
                            label=f'Rank {peak_rank + 1}: {peak_frequency.round(2)} Hz')

        fig.legend()
        fig.suptitle(f'Subject ID: {subject}, Age: {age}')
        rep.add_figs_to_section(fig, captions=subject, section=diagnosis)

        plt.close(fig)
        del fig

    rep.save(rep_name, open_browser=False, overwrite=True)






