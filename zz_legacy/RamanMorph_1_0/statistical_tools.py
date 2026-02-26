## STANDARD DEVIATION

def StandardDeviation(vals):

    import numpy as np

    summation = []

    for k in vals:

        summation.append((k - np.average(vals))**2)

    standard_deviation = ((1/(len(vals) - 1))*np.sum(summation))**(1/2)
    
    return standard_deviation
    
def moving_average(data, aver_half_window = 5, iteration = 1):
    
    import numpy as np
    
    it_count = 0
    
    while it_count != iteration:
        
        count = 0
        
        while count == 0:
            data_broadened = np.concatenate( # mirror-like broadening of spectra (he needs information behind
                (np.concatenate((            # edges of spectrum)
                    np.fliplr([data[0:3*aver_half_window]])[-1], data)), np.fliplr([data[-3*aver_half_window:]])[-1]))
            data_cummed = np.cumsum(data_broadened, dtype = float)
            data_cummed[aver_half_window:] = data_cummed[aver_half_window:] - data_cummed[:-aver_half_window]
            data_averaged = data_cummed[aver_half_window:] / aver_half_window
            data_averaged = np.array(data_averaged)[int((5*aver_half_window - 1)/2):-int((5*aver_half_window + 1)/2)]
            
            if len(data) != len(data_averaged): # control, peaks have to be symmetrical
                aver_half_window = aver_half_window - 1
            else:
                count += 1
                
        data = data_averaged
        it_count += 1
        
    return data
    
def area_computer(y_spec, peak_line, baseline, peak_chars):

    import numpy as np

    narrow_areas = [[] for k in range(len(peak_chars))]
    wide_areas = [[] for k in range(len(peak_chars))]

    for k in range(len(peak_chars)):

        for m in range(peak_chars[k][1], peak_chars[k][3] + 1):

            narrow_areas[k].append(y_spec[m] - peak_line[m])

        narrow_areas[k] = np.sum(narrow_areas[k])

   # condition, if two maxima in one peak, it doesn't have to compute it twice

        if k == 0 or not peak_chars[k][0] == peak_chars[k-1][0] and not peak_chars[k][4] == peak_chars[k-1][4]:

            for m in range(peak_chars[k][0], peak_chars[k][4] + 1):

                wide_areas[k].append(y_spec[m] - baseline[m])

            wide_areas[k] = np.sum(wide_areas[k])

        else:
            wide_areas[k] = wide_areas[k-1]

    #  narrow_areas, wide_areas = np.array(narrow_areas), np.array(wide_areas)

    areas = [[] for k in range(len(wide_areas))]

    for k in range(len(wide_areas)):
        for m in range(len(wide_areas)):

            if wide_areas[m] == wide_areas[k]:
                areas[k].append(m)

        if len(areas[k]) == 1:
            areas[k] = narrow_areas[k]

        else:
            areas[k] = np.sum(np.array(narrow_areas)[np.array(areas[k])])

        areas[k] = (narrow_areas[k]/areas[k])*wide_areas[k]

    areas = np.array(areas)
    
    area_indices = np.flip(np.argsort(areas), 0)

    areas_sum = np.sum(areas)

    area_portions = []

    for k in range(len(areas)):
        area_portions.append(areas[k]/areas_sum)

    area_portions = np.array(area_portions)

    area_output = [areas, area_portions, area_indices, areas_sum]

    return area_output
    
def ROCTest(prefix,name, area_output, peak_chars, limit_degrees = 45, ok_degrees = 30, picture = True):

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate

    areas = area_output[0]
    area_portions = area_output[1]
    area_indices = area_output[2]
    areas_sum = area_output[3]

    sorted_areas = areas[area_indices]
    sorted_area_portions = area_portions[area_indices]

    auc_part = 0

    # Sorting and normalizing areas and peak counts

    limit_tangent_point = np.tan(math.radians(limit_degrees))
    ok_tangent_point = np.tan(math.radians(ok_degrees))

    peak_order = []
    peak_normal_areas = []

    peak_order.append(0)
    peak_normal_areas.append(sorted_areas[0]/areas_sum)

    auc_part += peak_normal_areas[0]

    for k in range(1, len(peak_chars)):
        peak_order.append(float(k) / (len(peak_chars) - 1))
        peak_normal_areas.append(sorted_area_portions[k] + peak_normal_areas[-1])
        auc_part += peak_normal_areas[k]

    auc = auc_part/len(peak_order)

    # Interpolation of points
    
    if len(peak_order) > 3:
        tck = interpolate.splrep(peak_order, peak_normal_areas, s = 0)
    elif len(peak_order) == 3:
        tck = interpolate.splrep(peak_order, peak_normal_areas, s = 0, k = 2)
    elif len(peak_order) == 2:
        tck = interpolate.splrep(peak_order, peak_normal_areas, s = 0, k = 1)
    else:
        print("ROCTest: Something is wrong!")

    x_new = np.linspace(np.min(peak_order), np.max(peak_order), num=10*(len(peak_order)))
    y_new = interpolate.splev(x_new, tck, der = 0)

    # Derivation

    der = interpolate.splev(peak_order, interpolate.splder(tck))

    # classifying peaks (ok peaks, maybe peaks, no peaks)

    peak_classification = [[], [], []]

    for k in range(len(der)):
        if der[k] >= limit_tangent_point:
            peak_classification[0].append(k)
        elif limit_tangent_point > der[k] >= ok_tangent_point:
            peak_classification[1].append(k)
        elif der[k] < ok_tangent_point:
            peak_classification[2].append(k)
        else:
            print("ROCTest: Something is wrong!")

    ok_peaks = area_indices[peak_classification[0]]
    limit_peaks = area_indices[peak_classification[1]]
    no_peaks = area_indices[peak_classification[2]]

    classified_peaks = [ok_peaks, limit_peaks, no_peaks]

    if picture == True:

        fig, ax = plt.subplots(figsize = (5, 5))

        ax.plot(x_new, y_new, lw = 1)

        for k in peak_classification[0]:
            ax.plot(peak_order[k], peak_normal_areas[k], lw = 0, color = "green", marker = ".", ms = 10)
        for k in peak_classification[1]:
            ax.plot(peak_order[k], peak_normal_areas[k], lw = 0, color = "orange", marker = ".", ms = 10)
        for k in peak_classification[2]:
            ax.plot(peak_order[k], peak_normal_areas[k], lw = 0, color = "red", marker = ".", ms = 10) 

        ax.text(.6, .1, "AUC = " + '{:.2f}'.format(auc), fontsize = 15)
        
        #ax.tick_params(bottom = False, left = False)
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        fig.savefig(prefix+name+"_ROC.pdf", bbox_inches='tight')

    else:
        pass

    return peak_classification, peak_order, der, peak_normal_areas, auc
    
## NOISE

def noise_deviator(x_spec, y_spec, peak_chars, ok_peaks, maybe_peaks):

    import numpy as np
    from statistical_tools import StandardDeviation

    valid_peaks = []

    peak_chars = np.array(peak_chars)
    count = 0

    # REGIONS WITH ok AND maybe peaks

    for k in peak_chars:

        if k in peak_chars[ok_peaks] or k in peak_chars[maybe_peaks]:

            valid_peaks.append([])

            for m in range(len(k)):
                valid_peaks[count].append(k[m])
            count += 1

        else:
            pass

    # DEFINING NOISE REGIONS

    peak_chars_full = []
    noise_regions = []

    for k in range(len(valid_peaks)):
        if valid_peaks[k] == valid_peaks[0] and valid_peaks[k][0] != 0:                # Beginning of spectrum
            noise_regions.append([0,valid_peaks[k][0]])

        elif valid_peaks[k] == valid_peaks[-1] and valid_peaks[k][-1] != len(y_spec):  # End of spectrum
            noise_regions.append([valid_peaks[k-1][2],valid_peaks[k][0]])
            noise_regions.append([valid_peaks[k][-1],len(y_spec) - 1])

        else:
            noise_regions.append([valid_peaks[k-1][2],valid_peaks[k][0]])              # Middle part of spectrum

    # CONNECTING NOISE REGIONS        

    noise_regions_dim = []        

    for k in noise_regions:
        if not k[0] == k[1]:
            noise_regions_dim.append(k)
        else:
            pass
        
    # COMPUTING VALUES OF NOISE IN SPECIFIC POINTS

    x_noise, y_noise = [], []

    for k in range(len(noise_regions_dim)):
        ar_noise = np.arange(noise_regions_dim[k][0], noise_regions_dim[k][1])
        x_noise_help = []
        y_noise_help = []
        for m in range(1, len(ar_noise)):
            x_noise_help.append(x_spec[ar_noise[m]])
            y_noise_help.append(y_spec[ar_noise[m]] - y_spec[ar_noise[m] - 1] + np.amin(y_spec))
        x_noise.append(x_noise_help)
        y_noise.append(y_noise_help)



    if len(y_noise) != 0:
        vals = np.concatenate(y_noise)
    
    # COMPUTING STANDARD DEVIATION

        standard_deviation = StandardDeviation(vals)
    
    else:
        print("noise_deviator: Couldn't get enough data for computing standard deviation.")
        
        standard_deviation = None	
    
    return standard_deviation, x_noise, y_noise
