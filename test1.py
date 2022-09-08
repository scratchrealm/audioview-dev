# 9/8/22
# https://figurl.org/f?v=gs://figurl/spikesortingview-9&d=sha1://bbf7dede9fc901de56548af7925c39f10927b0b7&label=example%20audio

import numpy as np
import math
import kachery_cloud as kcl
import h5py
import spikeinterface.extractors as se
import sortingview.views as vv

def main():
    # Load or download the data from kachery
    fname_avi = kcl.load_file('sha1://a9fc5dc257e9a5c2e0af51114446bd39a44b8de0?label=2022_01_17_13_59_02_792530_cam_a.avi')
    fname_h5 = kcl.load_file('sha1://149e7e83682c3e0fbbef4dbb9153f469430464cb?label=mic_2022_01_17_13_59_02_792530.h5')

    # open the audio hdf5 file
    with h5py.File(fname_h5, 'r') as f:
        # print the items
        print('Items in hdf5 file')
        def print_item(name, obj):
            print(name, dict(obj.attrs))
        f.visititems(print_item)
        print('')

        # Load the audio data and prepare recording object
        sampling_frequency = 125000 # 125 KHz
        detect_threshold = 0.03 # for detecting events
        ch1 = np.array(f['ai_channels/ai0'])
        ch2 = np.array(f['ai_channels/ai1'])
        ch3 = np.array(f['ai_channels/ai2'])
        ch4 = np.array(f['ai_channels/ai3'])
        X = np.stack([ch1, ch2, ch3, ch4]).T
        maxval = np.max(X)
        print(f'Max. val = {maxval}')
        skip_duration_sec = 10 # skip first 10 seconds because of artifacts
        duration_sec = 60
        X = X[sampling_frequency * skip_duration_sec:sampling_frequency * (skip_duration_sec + duration_sec)]
        R = se.NumpyRecording(X, sampling_frequency=sampling_frequency)

        # Detect events and prepare raster plot - one unit per channel
        num_channels = X.shape[1]
        raster_plot_items = [
            vv.RasterPlotItem(unit_id=i, spike_times_sec=(detect_on_channel(
                data=X[:, i],
                detect_threshold=detect_threshold,
                detect_interval=int(sampling_frequency * 0.1),
                detect_sign=1
            ) / sampling_frequency).astype(np.float32))
            for i in range(num_channels)
        ]
        v_raster = vv.RasterPlot(
            start_time_sec=0,
            end_time_sec=R.get_total_duration(),
            plots=raster_plot_items
        )

        # Prepare raw traces plot
        v_raw = vv.RawTraces(
            start_time_sec=0,
            traces=R.get_traces(),
            sampling_frequency=R.sampling_frequency,
            channel_ids=[int(id) for id in R.get_channel_ids()]
        )

        view = vv.Box(
            direction='vertical',
            items=[
                vv.LayoutItem(v_raw),
                vv.LayoutItem(v_raster)
            ]
        )

        # get the figURL
        url = view.url(label='example audio')
        print(url)
        
        # view = vv.LiveTraces(recording=R, recording_id='test1')
        # view.run(label='test1', port=0)

# from mountainsort4
def detect_on_channel(data: np.ndarray, *, detect_threshold: float, detect_interval: float, detect_sign: int, margin: int=0):
    # Adjust the data to accommodate the detect_sign
    # After this adjustment, we only need to look for positive peaks
    if detect_sign < 0:
        data = data*(-1)
    elif detect_sign == 0:
        data = np.abs(data)
    elif detect_sign > 0:
        pass

    data = data.ravel()

    # An event at timepoint t is flagged if the following two criteria are met:
    # 1. The value at t is greater than the detection threshold (detect_threshold)
    # 2. The value at t is greater than the value at any other timepoint within plus or minus <detect_interval> samples

    # First split the data into segments of size detect_interval (don't worry about timepoints left over, we assume we have padding)
    N = len(data)
    S2 = math.floor(N / detect_interval)
    N2 = S2 * detect_interval
    data2 = np.reshape(data[0:N2], (S2, detect_interval))

    # Find the maximum on each segment (these are the initial candidates)
    max_inds2 = np.argmax(data2, axis=1)
    max_inds = max_inds2+detect_interval*np.arange(0, S2)
    max_vals = data[max_inds]

    # The following two tests compare the values of the candidates with the values of the neighbor candidates
    # If they are too close together, then discard the one that is smaller by setting its value to -1
    # Actually, this doesn't strictly satisfy the above criteria but it is close
    # TODO: fix the subtlety
    max_vals[np.where((max_inds[0:-1] >= max_inds[1:]-detect_interval)
                      & (max_vals[0:-1] < max_vals[1:]))[0]] = -1
    max_vals[1+np.array(np.where((max_inds[1:] <= max_inds[0:-1] +
                        detect_interval) & (max_vals[1:] <= max_vals[0:-1]))[0])] = -1

    # Finally we use only the candidates that satisfy the detect_threshold condition
    times = max_inds[np.where(max_vals >= detect_threshold)[0]]
    if margin > 0:
        times = times[np.where((times >= margin) & (times < N-margin))[0]]

    return times

if __name__ == '__main__':
    main()