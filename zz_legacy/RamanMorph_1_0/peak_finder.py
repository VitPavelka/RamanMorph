## PEAK CHARACTERIZATION
## FINDING TIPS AND TAILS

def peak_characterization(y_spec, y_eroded, candidates, hwc):

    import numpy as np	

    ## PRODUCTION OF PEAK CHARACTERIZATION LISTS

    tip_candidates = list(candidates[0])
    tail_candidates = list(candidates[1])
    base_candidates = list(candidates[2])

    # append edges of spectrum (number of elements in each list has to be the same)

    new_tail_candidates = [0]
    new_base_candidates = [0]

    if 0 not in tail_candidates:
        [new_tail_candidates.append(k) for k in tail_candidates]
        tail_candidates = new_tail_candidates

    if 0 not in base_candidates:
        [new_base_candidates.append(k) for k in base_candidates]
        base_candidates = new_base_candidates

    if len(y_spec) - 1 not in tail_candidates:
        tail_candidates.append(len(y_spec) - 1)

    if len(y_spec) - 1 not in base_candidates:
        base_candidates.append(len(y_spec) - 1)

    peak_chars = [ [] for _ in range(5) ]

    # characterization of narrow peaks (candidates from erosion w1 = half_window_narrow)

    for k in range(len(tip_candidates)):
        for m in range(1, len(tail_candidates)):
            if tail_candidates[m - 1] < tip_candidates[k] and tail_candidates[m] > tip_candidates[k]:

                right = tail_candidates[m-1]
                tip = tip_candidates[k]
                left = tail_candidates[m]

                peak_chars[1].append(right)
                peak_chars[2].append(tip)
                peak_chars[3].append(left)

                break

    # characterization of wide peaks (candidates from baseline w2 = half_window_wide) if w1 != w2

    if hwc == True:

        for k in range(len(tip_candidates)):
            for m in range(1, len(base_candidates)):
                if base_candidates[m - 1] < tip_candidates[k] and base_candidates[m] > tip_candidates[k]:

                    base_right = base_candidates[m-1]
                    base_left = base_candidates[m]

                    peak_chars[0].append(base_right)
                    peak_chars[4].append(base_left)
     
    else:

        peak_chars[0] = peak_chars[1]
        peak_chars[4] = peak_chars[3]

    peak_chars = np.array(peak_chars).T

    ## FINDING MINIMA IN MULTIPEAKS and DELETING ABUNDANT ONES

    new_borders = [peak_chars[0][1]]

    for k in range(1, len(peak_chars)):
        if peak_chars[k][1] == peak_chars[k-1][1]:

            diff_erosion_spectrum = []

            for m in range(peak_chars[k - 1][2], peak_chars[k][2]):
                diff_erosion_spectrum.append(y_spec[m] - y_eroded[m])

            new_border = np.argmin(diff_erosion_spectrum) + peak_chars[k - 1][2]
            peak_chars[k - 1][3] = new_border

            new_borders.append(np.argmin(diff_erosion_spectrum) + peak_chars[k - 1][2])

        else:
            new_borders.append(peak_chars[k][1])

    if len(peak_chars) == len(new_borders):
        for k in range(len(peak_chars)):

            peak_chars[k][1] = new_borders[k]

    tt_output = peak_chars.T

    return peak_chars, tt_output[0], tt_output[1], tt_output[2], tt_output[3], tt_output[4]
    
## DERIVING PEAK LINE
    
def peak_line_derivation(x_spec, y_spec, y_eroded, peak_chars, iteration_max = 20):
    
    from morphological_operations import improved_linear_interpolation

    edges = []
    slopes = []
    peak_line = []

    c = 0

    edges.append(peak_chars[0][1])

    for k in range(1, len(peak_chars)):
        if peak_chars[k][1] != edges[-1]: 
            edges.append(peak_chars[k][1])
        else:
            pass

        if peak_chars[k][3] != edges[-1]:
            edges.append(peak_chars[k][3])

    for k in range(1, len(edges)):

        dy = y_spec[edges[k]] - y_spec[edges[k - 1]]
        dx = x_spec[edges[k]] - x_spec[edges[k - 1]]
        slopes.append(dy/dx)

    for k in range(len(x_spec)):
        if k < edges[0]:             # edge of spectrum replaced by eroded y, beware, you have more possibilities

            peak_line.append(y_eroded[k])
        elif edges[c] <= k and k < edges[c + 1]:

            peak_line.append((x_spec[k] - x_spec[edges[c]])*slopes[c] + y_spec[edges[c]])

        elif k > edges[-1]:

            peak_line.append(y_eroded[k])

        elif k == edges[c + 1]:

            peak_line.append(y_spec[k])

            if c + 1 != len(slopes):
                c += 1

        else:
            print("peak_line: Something is wrong!")
    
    peak_line = improved_linear_interpolation(x_spec, y_spec, peak_line, iteration_max = iteration_max)
    
    return peak_line
    
##  CORRECTION OF TAILS

def tail_correction(peak_chars, y_spec, baseline, peak_line):

    for k in peak_chars:

        # baseline tails

        for m in range(k[0], k[2]):

            if y_spec[m] == baseline[m] and y_spec[m] == peak_line[m]:

                k[0] = m

        for m in range(k[2], k[4]):

            if y_spec[m] == baseline[m] and y_spec[m] == peak_line[m]:

                k[4] = m
                break

      #   peakline tails

        for m in range(k[1],k[2]):

            if y_spec[m] == peak_line[m]:

                k[1] = m

        for m in range(k[2], k[3]):

            if y_spec[m] == peak_line[m]:

                k[3] = m
                break 
                
    return peak_chars