# Version 1.0

import time
import sys
import math

import numpy as np

import data_utilizers as du
import morphological_operations as mo
import peak_finder as pf
import statistical_tools as st

suffix, limit_degrees, ok_degrees, prefix, datafiles, w1, w2, figure, fig_suffix, delimiter,comment = du.data_configuration()

half_window_narrow = w1
half_window_wide = w2
count = 0

start_great_time = time.time()

for datafile in datafiles.split(','):
    
    count += 1

    start_time = time.time()
    x_spec, y_spec = du.data_input(prefix, datafile, suffix,delimiter,comment)  # data loading

    ## PASS I
    y_dila, tip_candidates = mo.dilation(y_spec, half_window_narrow)    # morphological filtering
    y_eros, tail_candidates = mo.erosion(y_spec, half_window_narrow)
    eroded_base, base_candidates = mo.erosion(y_spec, half_window_wide)
       
    candidates = (tip_candidates, tail_candidates, base_candidates)    # candidates of peak tips and tails

    peak_chars, base_right_tails, right_tails, tips, left_tails, base_left_tails = pf.peak_characterization(y_spec, y_eros, candidates, True)           # splitting multipeaks
    peak_chars = np.array(peak_chars)

    ## PASS II
    peak_line = pf.peak_line_derivation(x_spec, y_spec, y_eros, peak_chars)    # improved peakline
    baseline = mo.improved_linear_interpolation(x_spec, y_spec, eroded_base)  # improved baseline
    baseline = mo.improved_linear_interpolation(x_spec, peak_line, baseline)  # improved baseline with respect to peakline
    
    peak_line_output = np.array([x_spec, peak_line]).T
    baseline_output = np.array([x_spec, baseline]).T
    
    lines = [x_spec, y_spec, peak_line, baseline]
    
    peak_chars = pf.tail_correction(peak_chars, y_spec, baseline, peak_line)    #correction of peak tails
    
    ##POSTPROCESSING
    
    chars, peak_output, std, auc = du.data_processor(prefix, datafile, lines, peak_chars, limit_degrees, ok_degrees)

    # data output

    du.data_output(prefix, datafile, lines, chars, peak_line_output, baseline_output, peak_output, std, auc, delimiter, figure, fig_suffix)
    
    end_time = time.time() 

    print("RamanMorph: "+str(datafile)+" Done in "+str(du.time_count(start_time,end_time))+", count: "+str(count))
    
end_great_time = time.time()
print("RamanMorph: Total time: "+str(du.time_count(start_great_time, end_great_time))+".")