from BobAnalysis.core.session import Session
from BobAnalysis.core.metadata import oeid_csid_to_index
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

def smooth(x,window_len=11,window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode=mode)
    return y

def main(dff_trace):

    k_min = 0
    k_max = 10
    delta = 3

    dff_trace = smooth(dff_trace, 5)

    var_dict = {}

    for ii in range(len(dff_trace)):

        if ii + k_min >= 0 and ii + k_max <= len(dff_trace):
            trace = dff_trace[ii + k_min:ii + k_max]

            xx = (trace - trace[0])[delta] - (trace - trace[0])[0]
            # yy = max((trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2],
            #          (trace - trace[0])[delta + 3] - (trace - trace[0])[0 + 3],
            #          (trace - trace[0])[delta + 4] - (trace - trace[0])[0 + 4])
            yy = (trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2]#,
                     # (trace - trace[0])[delta + 3] - (trace - trace[0])[0 + 3],
                     # (trace - trace[0])[delta + 4] - (trace - trace[0])[0 + 4])


            var_dict[ii] = (trace[0], trace[-1], xx, yy)

    xx_list, yy_list = [], []
    for _, _, xx, yy in var_dict.itervalues():
        xx_list.append(xx)
        yy_list.append(yy)


    mu_x = np.median(xx_list)
    mu_y = np.median(yy_list)

    xx_centered = np.array(xx_list)-mu_x
    yy_centered = np.array(yy_list)-mu_y

    std_factor = 1
    std_x = 1./std_factor*np.percentile(np.abs(xx_centered), [100*(1-2*(1-sps.norm.cdf(std_factor)))])
    std_y = 1./std_factor*np.percentile(np.abs(yy_centered), [100*(1-2*(1-sps.norm.cdf(std_factor)))])

    curr_inds = []
    allowed_sigma = 4
    for ii, (xi, yi) in enumerate(zip(xx_centered, yy_centered)):
        if np.sqrt(((xi)/std_x)**2+((yi)/std_y)**2) < allowed_sigma:
            curr_inds.append(True)
        else:
            curr_inds.append(False)

    curr_inds = np.array(curr_inds)
    data_x = xx_centered[curr_inds]
    data_y = yy_centered[curr_inds]
    Cov = np.cov(data_x, data_y)
    Cov_Factor = np.linalg.cholesky(Cov)
    Cov_Factor_Inv = np.linalg.inv(Cov_Factor)

    #===================================================================================================================

    p2 = True

    fig, ax = plt.subplots()
    if p2:
        fig2, ax2 = plt.subplots()
        ax2.plot(dff_trace)
    noise_threshold = max(allowed_sigma * std_x + mu_x, allowed_sigma * std_y + mu_y)
    mu_array = np.array([mu_x, mu_y])
    yes_set_1, yes_set_2, no_set, yes_set_1_u, yes_set_1_l = set(), set(), set(), set(), set()
    for ii, (t0, tf, xx, yy) in var_dict.iteritems():


        xi_z, yi_z = Cov_Factor_Inv.dot((np.array([xx,yy]) - mu_array))

        # Conditions in order:
        # 1) Outside noise blob
        # 2) Minimum change in df/f
        # 3) Change evoked by this trial, not previous
        # 4) At end of trace, ended up outside of noise floor

        if np.sqrt(xi_z**2 + yi_z**2) > 3:# and yy > .05 and xx < yy and tf > noise_threshold/2:
            if yy > -xx:
                if yy > xx:
                    yes_set_1_u.add(ii)
                    ax.plot([xx], [yy], 'r.')
                else:
                    yes_set_1_l.add(ii)
                    ax.plot([xx], [yy], 'g.')
            else:
                yes_set_2.add(ii)
                ax.plot([xx], [yy], 'k.')
        else:
            no_set.add(ii)
            ax.plot([xx], [yy], 'b.')

    if p2:
        yes_list = list(yes_set_1_u)
        ax2.plot(yes_list, dff_trace[np.array(yes_list)], 'ro')

        yes_list = list(yes_set_1_l)
        ax2.plot(yes_list, dff_trace[np.array(yes_list)], 'go')

        yes_list = list(yes_set_2)
        ax2.plot(yes_list, dff_trace[np.array(yes_list)], 'ko')

    plt.show()

class EventDetection(object):

    def __init__(self, **kwargs):

        self.session = Session(**kwargs)

    def get_dff(self, csid=None):

        if csid is None:
            return self.session.get_dff_array(self.session.data, self.session.oeid)
        else:
            index = oeid_csid_to_index(self.session.oeid, csid)
            return self.session.get_dff_array(self.session.data, self.session.oeid)[index, :]


if __name__ == "__main__":

    E = EventDetection(oeid=510536157)

    dff = E.get_dff(517504593)

    main(dff[:10000])

    # plt.plot()
    plt.show()
