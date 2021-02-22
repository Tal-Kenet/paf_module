# savitzky-golay filter frame width (in odd-numbered samples), longer = more smoothing
sgf_width = 11

# savitzky-golay filter polynomial degree/order, greater = less smoothing, better preservation of feature heights, widths
k = 5 # default 5

# alpha band width [Hz]
alpha_band = [7, 13]

# PSD frequency resolution preferences
n_fft = 1024
desired_freq_res = 0.25 # Hz

# difference in peak height by which highest peak candidate must exceed any competitors
diffp = 0.2 # 20%

# minimum number of channel estimates for computing cross-channel averages
cmin = 3

# number of unique channels to use for ranking
n_peaks_to_rank = 5
rank_by_colors = ['b', 'g', 'r', 'm', 'y']

# channels to use for PAF - subselection of occipital channels
occipital_subselection_legacy = ['MEG 1942', 'MEG 1943', 'MEG 1732', 'MEG 1733', 'MEG 1932', 'MEG 1933', 'MEG 1922',
                                 'MEG 1923', 'MEG 2112', 'MEG 2113', 'MEG 2122', 'MEG 2123', 'MEG 2322', 'MEG 2323',
                                 'MEG 2512', 'MEG 2513', 'MEG 2332', 'MEG 2333', 'MEG 2342', 'MEG 2343']

occipital_subselection_modern = [ocss.replace(' ', '') for ocss in occipital_subselection_legacy]