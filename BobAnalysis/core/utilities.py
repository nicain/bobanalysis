import numpy as np

def get_sparse_noise_epoch_mask_list(st, number_of_acquisition_frames, threshold=7):

    delta = (st.start.values[1:] - st.end.values[:-1])
    cut_inds = np.where(delta > threshold)[0] + 1

    epoch_mask_list = []

    if len(cut_inds) > 2:
        warnings.warn('more than 2 epochs cut')
        print '    ', len(delta), cut_inds

    for ii in range(len(cut_inds)+1):

        if ii == 0:
            first_ind = st.iloc[0].start
        else:
            first_ind = st.iloc[cut_inds[ii-1]].start

        if ii == len(cut_inds):
            last_ind_inclusive = st.iloc[-1].end
        else:
            last_ind_inclusive = st.iloc[cut_inds[ii]-1].end

        # curr_epoch_mask = np.zeros(number_of_acquisition_frames, dtype=np.bool)
        # curr_epoch_mask[first_ind:last_ind_inclusive+1] = True
        epoch_mask_list.append((first_ind,last_ind_inclusive))

    return epoch_mask_list