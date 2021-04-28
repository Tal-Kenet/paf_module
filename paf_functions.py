import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import savgol_filter

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
    except ValueError: # older data uses different channel-naming convention
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=alpha_window[0], fmax=alpha_window[1], n_fft=fft_length,
                                                  n_jobs=n_jobs, picks=picks_legacy)
    del raw
    return psd, freqs


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


def find_channel_peak_freq_and_qweight(freqs, signal, peak_candidate_inds, peak_candidate_bounds, freq_res, diffp):
    """
    wrapper function for gather_peak_candidate_metrics() below
    :param freqs: frequency array
    :param signal: filtered channel PSD array
    :param peak_inds: indices of peaks
    :param peak_bounds: nested arrays containing: minimia widths, minima heights, left and right interpolated indices
    :param freq_res: frequency resolution
    :return: peak frequencies and their corresponding Q-weights for an entire channel
    """
    channel_freq_candidates = []
    channel_height_candidates = []
    channel_qweight_candidates = []

    # collect data on each peak candidate...
    for i in range(peak_candidate_inds.size):
        freq_candidate, height_candidate, q_candidate = gather_peak_candidate_metrics(i, freqs, signal,
                                                                                      peak_candidate_inds,
                                                                                      peak_candidate_bounds, freq_res)
        channel_freq_candidates.append(freq_candidate)
        channel_height_candidates.append(height_candidate)
        channel_qweight_candidates.append(q_candidate)
    # remove peak candidates that are not viable PAF
    channel_peak_freqs, channel_peak_heights, channel_peak_qweights = eliminate_peak_candidates(channel_freq_candidates,
                                                                                                channel_height_candidates,
                                                                                                channel_qweight_candidates, diffp)
    return channel_peak_freqs, channel_peak_qweights


def gather_peak_candidate_metrics(iter, freqs, signal, peak_inds, peak_bounds, freq_res):
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

    return peak_freq, peak_height, peak_q


def eliminate_peak_candidates(freq_candidates, height_candidates, qweight_candidates, diffp):
    """
    use diffP parameter to eliminate peak candidates
    :param peak_freqs: list of peak frequencies in Hz
    :param peak_heights: list of peak heights in PSD amplitude
    :param peak_qweights: list of peak Q-weights
    :param diffp: float denoting percentage difference in peak height that a peak candidate must be greater than others
    :return:
    """
    candidate_indices_to_remove = []
    for idx, amp in enumerate(height_candidates):
        adjusted_amp = amp * (1 - diffp)
        other_peak_candidates = np.delete(height_candidates, idx)
        peak_candidate_viability = np.where(adjusted_amp > other_peak_candidates)[0]
        if peak_candidate_viability.size == 0:
            candidate_indices_to_remove.append(idx)

    peak_freqs = np.delete(freq_candidates, candidate_indices_to_remove)
    peak_heights = np.delete(height_candidates, candidate_indices_to_remove)
    peak_qweights = np.delete(qweight_candidates, candidate_indices_to_remove)
    return peak_freqs, peak_heights, peak_qweights


def determine_channel_weights_and_rank_peaks_by_qweight(peak_qweights, peak_freqs):
    """
    rank peak candidates in descending order (high to low) by their Q-weight (max Q-weight normalized)
    :param channel_weights: normalized Q-weights for each channel's peak alpha frequency
    :param peak_freqs: peak frequencies for each channel
    :return: unique, sorted (descending order) peak frequencies for a subject's channels
    """
    if peak_qweights.max() > 0: # check if any PAF was found...
        channel_weights = peak_qweights / peak_qweights.max()  # perform cross-channel averaging
        sorted_channel_weights = np.argsort(channel_weights)[::-1]
        sorted_peak_freqs = peak_freqs[sorted_channel_weights] # peak frequencies
        _, uniques_idx = np.unique(sorted_peak_freqs, return_index=True)
        sorted_unique_peak_freqs = sorted_peak_freqs[np.sort(uniques_idx)]
    else:
        channel_weights = None
        sorted_unique_peak_freqs = None

    return channel_weights, sorted_unique_peak_freqs


def make_figure_for_report(freqs, psd, smooth_psd, ranked_peak_freqs, colors, subject, age):

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
    if ranked_peak_freqs:
        for peak_rank, peak_frequency in enumerate(ranked_peak_freqs):
            ax_orig.axvline(peak_frequency, ls='--', c=colors[peak_rank],
                            label=f'Rank {peak_rank + 1}: {peak_frequency.round(2)} Hz')

    fig.legend()
    fig.suptitle(f'Subject ID: {subject}, Age: {age}')
    return fig