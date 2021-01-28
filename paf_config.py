
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