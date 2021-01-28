import paf_config as paf_cfg
#import paradigm_config as para_cfg
import mne
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
import pandas as pd
import fnmatch
import os
import matplotlib.pyplot as plt
import time

def gather_peak_metrics(iter, freqs, signal, peak_inds, peak_bounds, delta_freq):
    """
    determine frequency span of each peak, as well as integrating the region bound by the peak's extrema
    :param iter: current index value to access location of peak
    :param freqs: frequency array
    :param signal: filtered channel PSD array
    :param peak_inds: indices of peaks
    :param peak_bounds: nested arrays containing: minimia widths, minima heights, left and right interpolated indices
    :param delta_freq: frequency resolution
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
    left_minima_freq = freqs[peak_idx] - (delta_freq * (peak_idx - left_minima_sample))
    right_minima_freq = freqs[peak_idx] + np.abs(delta_freq * (peak_idx - right_minima_sample))

    # aggregate the extrema
    extrema_freqs = np.array([left_minima_freq, peak_freq, right_minima_freq])
    extrema_heights = np.array([minima_height, peak_height, minima_height])

    # final metrics...
    au_peak = np.trapz(extrema_heights, x=extrema_freqs)  # find the area under the peak (AUP)...
    peak_q = au_peak / (delta_freq * minima_width)  # scale by the frequency span of the peak

    return peak_freq, peak_q


def find_channel_peak_freq_and_qweight(freqs, signal, peak_inds, peak_bounds, delta_freq):
    """
    wrapper function for gather_peak_metrics() above
    :param freqs: frequency array
    :param signal: filtered channel PSD array
    :param peak_inds: indices of peaks
    :param peak_bounds: nested arrays containing: minimia widths, minima heights, left and right interpolated indices
    :param delta_freq: frequency resolution
    :return: peak frequencies and their corresponding Q-weights for an entire channel
    """
    channel_peak_freqs = []
    channel_peak_qweights = []

    # collect data on each peak...
    for i in range(peak_inds.size):
        peak_freq, peak_q = gather_peak_metrics(i, freqs, signal, peak_inds, peak_bounds, delta_freq)

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


def compute_psd(evoked, picks1, picks2, n_jobs, alpha_window, cropped=True):
    """
    :param evoked: mne.Evoked object
    :param picks: selection of channels to parse
    :param alpha_window: lower and upper bounds to frequency range spanning alpha oscillations
    :param cropped: if True, crop the frequency and PSD array returned to the range of alpha_window
    :return: PSD array (n_channels, n_freqs), frequency array (n_freqs,)
    """
    try:
        psd, freqs = mne.time_frequency.psd_welch(evoked, n_fft=3300, picks=picks1, n_jobs=n_jobs)
    except:
        psd, freqs = mne.time_frequency.psd_welch(evoked, n_fft=3300, picks=picks2, n_jobs=n_jobs)
    if cropped:
        alpha_low_bound_idx = np.where(freqs >= alpha_window[0])[0][0]
        alpha_hi_bound_idx = np.where(freqs >= alpha_window[1])[0][0]
        freqs = freqs[alpha_low_bound_idx:alpha_hi_bound_idx]
        psd = psd[:, alpha_low_bound_idx:alpha_hi_bound_idx]

    return psd, freqs


def run_paf(evoked): # main body that interacts with sensor_space_analysis module

    # select occipital channels only for this resting state analysis
    occipital_selection1 = mne.read_selection(['Left-occipital', 'Right-occipital'])
    occipital_selection2 = [ocs.replace(' ', '') for ocs in occipital_selection1]

    psd, freqs = compute_psd(evoked, occipital_selection2, occipital_selection1, n_jobs=18, alpha_window=paf_cfg.alpha_band)
    delta_freq = np.diff(freqs).mean() # change this variable name to something else (delta waves exist)
    print(psd.shape)
    print(freqs)
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
        peak_bounds = peak_widths(signal_filtered, peak_inds, rel_height=0.5) # return minima characteristics for each peak

        channel_peak_freqs, channel_peak_qweights = find_channel_peak_freq_and_qweight(freqs, signal_filtered, peak_inds, peak_bounds, delta_freq)

        # determine the most significant peak in the channel
        channel_peak_freq = channel_peak_freqs[np.argmax(channel_peak_qweights)]
        channel_peak_qweight = channel_peak_qweights[np.argmax(channel_peak_qweights)]

        # append these to the subject-wide list...
        subject_peak_freqs.append(channel_peak_freq)
        subject_peak_qweights.append(channel_peak_qweight)

    # perform cross-channel averaging
    channel_weights = np.array(subject_peak_qweights) / np.array(subject_peak_qweights).max()
    # not so sure about the below... it seems very passive and average-y...
    paf = np.sum(np.array(subject_peak_freqs) * channel_weights) / np.sum(channel_weights)
    # why not locate the greatest Q?
    paf_max_q = subject_peak_freqs[np.argmax(subject_peak_qweights)]
    return freqs, psd, smooth_psd, subject_peak_freqs, channel_weights, paf, paf_max_q




if __name__ == '__main__':
    #evos = sorted(fnmatch.filter(os.listdir(), '*ave.fif'))
    #first_pass = mne.read_evokeds(evos[0])[0]
    #second_pass = mne.read_evokeds(evos[1])[0]
    start = time.time()
    rep = mne.Report()
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

        evo_match = fnmatch.filter(os.listdir(), f'{subject}*ave.fif')
        if len(evo_match) == 0:
            continue
        subject_evo = mne.read_evokeds(evo_match[0])[0]

        freqs, psd, smooth_psd, subject_peak_freqs, channel_weights, paf, paf_max_q = run_paf(subject_evo)
        #print(f'Subject {subject}, PAF (max Q): {paf_max_q}')
        fig, (ax_orig, ax_smooth) = plt.subplots(2, 1, sharex=True, sharey=True)
        for channel_orig in psd:
            ax_orig.plot(freqs, channel_orig)
        ax_orig.axvline(paf_max_q, ls='--', c='r', label='Max Q')
        ax_orig.axvline(paf, ls='--', c='k', label='MATLAB')
        #ax_orig.set_xlabel('Frequency [Hz]')
        ax_orig.set_ylabel('PSD (original)')
        ax_orig.set_ylim((0, psd.max() + 1.5))
        #ax_orig.set_title('Original PSD')
        
        for channel_smooth in smooth_psd:
            ax_smooth.plot(freqs, channel_smooth)
        ax_smooth.axvline(paf_max_q, ls='--', c='r', label='Max Q')
        ax_smooth.axvline(paf, ls='--', c='k', label='MATLAB')
        ax_smooth.set_xlabel('Frequency [Hz]')
        ax_smooth.set_ylabel('Smoothed PSD')
        ax_smooth.set_ylim((0, psd.max() + 1.5))
        #ax_smooth.set_title('Filtered PSD')
        
        fig.legend()
        fig.suptitle(f'Subject ID: {subject}, Age: {age}')
        rep.add_figs_to_section(fig, captions=subject, section=diagnosis)
        """
        fig = plt.figure()
        ax = fig.gca()
        for channel in psd:
            ax.plot(freqs, channel)
        ax.axvline(paf_max_q, ls='--', c='r', label='Max Q')
        ax.axvline(paf, ls=':', c='k', label='MATLAB')
        ax.set_ylabel('PSD')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylim((0, psd.max() + 1.5))
        fig.legend()
        fig.suptitle(f'Subject ID: {subject}, Age: {age}')
        rep.add_figs_to_section(fig, captions=subject, section=diagnosis)
        """
        plt.close(fig)
        del fig
    rep.save('paf_vs_age_fixation.html', open_browser=False, overwrite=True)
    print(f'Time to complete: {time.time() - start}')
        
        
        
    """
    fig = plt.figure()
    ax = fig.gca()
    for channel in psd:
        ax.plot(freqs, channel)
    ax.axvline(paf_max_q, ls='--', label='PAF')
    ax.set_ylabel('PSD (pre-cleaning)')
    ax.set_xlabel('Frequency [Hz]')
    fig.legend()
    fig.show()
    """











