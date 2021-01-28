from mne import read_selection
# savitzky-golay filter frame width (in odd-numbered samples), longer = more smoothing
sgf_width = 11

# savitzky-golay filter polynomial degree/order, greater = less smoothing, better preservation of feature heights, widths
k = 5 # default 5

# alpha band width [Hz]
alpha_band = [7, 13]

# difference in peak height by which highest peak candidate must exceed any competitors
diffp = 0.2 # 20%

# minimum number of channel estimates for computing cross-channel averages
cmin = 3

# channels to use for PAF
occipital_selection_all_legacy = read_selection(['Left-occipital', 'Right-occipital'])
occipital_selection_all = [ocs.replace(' ', '') for ocs in occipital_selection_all_legacy]

# narrowed occipital channels
occipital_selection_narrow_legacy = ['MEG 1941', 'MEG 1942', 'MEG 1943', 'MEG 1921', 'MEG 1922', 'MEG 1923',
                              'MEG 1931', 'MEG 1932', 'MEG 1933', 'MEG 1731', 'MEG 1732', 'MEG 1733',
                              'MEG 2331', 'MEG 2332', 'MEG 2333', 'MEG 2341', 'MEG 2342', 'MEG 2343',
                              'MEG 2511', 'MEG 2512', 'MEG 2513', 'MEG 2321', 'MEG 2322', 'MEG 2323']
occipital_selection_narrow = [nocs.replace(' ', '') for nocs in occipital_selection_narrow_legacy]