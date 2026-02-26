def time_count(beg,end):

    interval=end-beg 
    hours, remainder = divmod(interval, 3600)
    minutes, rem2 = divmod(remainder, 60)
    seconds,ms = divmod(rem2,1)

    out = ''

    if int(hours)>0:
        out = out + str(int(hours))+' h '
    if int(minutes)>0:
        out = out + str(int(minutes))+' min '
    if int(seconds)>0:
        out = out + str(int(seconds))+' s '

    out = out+ str(int(1000*ms))+' ms'
    return out

def data_configuration():

    import configparser
#    import ConfigParser
    import ast
    import sys

    def ConfigSectionMap(section):
        dict1 = {}
        option = Config.options(section)
        for option in options:
            try:
                dict1[option] = Config.get(section, option)
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    Config = configparser.ConfigParser()
#    Config = ConfigParser.ConfigParser()

    Config.read("RamanMorph.ini")

    job = "common"
    
    suffix = ".txt"
    limit_degrees = 45
    ok_degrees = 30

    options = Config.options(job)
    
    if "suffix" in options:
        suffix = ConfigSectionMap(job)["suffix"]
        if suffix == "":
            suffix = ".txt"
            
    if "fig_suffix" in options:
        fig_suffix = ConfigSectionMap(job)["fig_suffix"]
        if fig_suffix == "":
            fig_suffix = ".pdf"

    if "limit_degrees" in options:
        limit_degrees = int(ConfigSectionMap(job)["limit_degrees"])
        if limit_degrees == "":
            limit_degrees = 45

    if "ok_degrees" in options:
        ok_degrees = int(ConfigSectionMap(job)["ok_degrees"])
        if ok_degrees == "":
            ok_degrees = 30

    job = sys.argv[1]
    
    prefix = ""
    datafile = ""
    w1 = 10
    w2 = 30

    options = Config.options(job)

    if "prefix" in options:
        prefix = ConfigSectionMap(job)["prefix"]
            
    if "datafile" in options:
        datafile = ConfigSectionMap(job)["datafile"]
        if datafile == "":
        
            import glob
            
            datafile = list(glob.glob(prefix+"*.txt"))
            sep_list = ["_peaks", "_parameters", "_baseline", "_peakline"]
            datafile = [x.replace(sep_list[0],"").replace(sep_list[1],"").replace(sep_list[2],"").replace(sep_list[3],"") for x in datafile]
            datafile = list(set(datafile))
            datafile = [x.replace(".txt","").replace(prefix, "") for x in datafile]
            split_delim = ","
            datafile = split_delim.join(datafile)
        
    if "w1" in options:
        w1 = int(ConfigSectionMap(job)["w1"])
        if w1 == "":
            w1 = 10
    
    if "w2" in options:
        w2 = int(ConfigSectionMap(job)["w2"])
        if w2 == "":
            w2 = 30
    
    if "figure" in options:
        figure = Config.getboolean(job, "figure")
        if figure == "":
            figure = False

    delimiter='\t'
    if "delimiter" in options:
        delimiter = ConfigSectionMap(job)["delimiter"]
        if delimiter == "":
            delimiter = None

    comment='#'
    if "comment" in options:
        comment = ConfigSectionMap(job)["comment"]
        if comment == "":
            comment = '#'

    return suffix, limit_degrees, ok_degrees, prefix, datafile, w1, w2, figure, fig_suffix, delimiter, comment

def data_input(prefix, name, suffix,delim,comm):

    import pandas as pd
    import numpy as np
    
    upload_name = prefix + name + suffix

    data = pd.read_table(upload_name, header=None,delimiter=delim,comment=comm)

    x_spec = np.array(data[0])
    y_spec = np.array(data[1])
    
    return x_spec, y_spec
    
def data_processor(prefix, datafile, lines, peak_chars, limit_degrees, ok_degrees):
    
    import math
    import numpy as np
    import statistical_tools as st
    
    x_spec = lines[0]
    y_spec = lines[1]
    peak_line = lines[2]
    baseline = lines[3]
    
    # computing intensites (areas) of found peaks
    area_output = st.area_computer(lines[1], lines[2], lines[3], peak_chars)
    
    areas = area_output[0]
    area_portions = area_output[1]
    area_indices = area_output[2]
    areas_sum = area_output[3]
    
    # sorting of peaks (valid - qualitatively valid - invalid)
    peak_triad = peak_chars.T[[1,2,3]].T

    peak_classification, peak_order, der, cumul_area, auc = st.ROCTest(prefix, datafile, area_output, peak_triad, limit_degrees = limit_degrees, ok_degrees = ok_degrees, picture = False)

    ok_peaks = area_indices[peak_classification[0]]
    maybe_peaks = area_indices[peak_classification[1]]
    no_peaks = area_indices[peak_classification[2]]
    
    chars = [peak_chars, ok_peaks, maybe_peaks, no_peaks]
    
    # spectrum parameters

    std, x_noise, y_noise = st.noise_deviator(lines[0], lines[1], peak_triad, ok_peaks, maybe_peaks)
    
    # sorting of arrays for data output

    wavelengths = x_spec[peak_chars].T

    for k in range(len(wavelengths)):
        wavelengths[k] = wavelengths[k][area_indices]

    peak_intensities = areas[area_indices]
    absolute_peak_heights = y_spec[np.array(peak_triad).T[1]][area_indices]
    peak_heights = (y_spec[np.array(peak_triad).T[1]] - np.array(peak_line)[np.array(peak_triad).T[1]])[area_indices]
    peak_baseline = (y_spec[np.array(peak_triad).T[1]] - np.array(baseline)[np.array(peak_triad).T[1]])[area_indices]

    angles = []

    for k in der:
        angles.append(math.degrees(np.arctan(k)))

    peak_output = np.array([wavelengths[0], wavelengths[1], wavelengths[2], wavelengths[3], wavelengths[4], peak_intensities, peak_heights, peak_baseline, absolute_peak_heights, cumul_area, angles]).T
    
    return chars, peak_output, std, auc

def data_output(prefix, datafile, lines, chars, peak_line_output, baseline_output, peak_output, std, auc, delim, figure, fig_suffix):

    import numpy as np

    peak_params_header = ["nu(rightBL) ", " nu(rightPL) ", " nu(max) ", " nu(leftPL) ", " nu(leftBL) ", " peak area(A) ", " Imax-Pl ", " Imax-Bl ", 
                          " Imax ", " A(cummulative) ", " ROC angles "]
    parameters_output = np.array([["noise: ", std, "", "", "", "", "", "", "", "", ""], 
                                  ["auc: " , auc, "", "", "", "", "", "", "", "", ""], 
                                  ["Peak Parameters:", "", "", "", "", "", "", "", "", "", ""], 
                                  peak_params_header])

    np.savetxt(prefix+datafile+"_peakline.txt", peak_line_output, delimiter=delim)
    np.savetxt(prefix+datafile+"_baseline.txt", baseline_output, delimiter=delim)
    np.savetxt(prefix+datafile+"_peaks.txt", peak_output, fmt=['%1.1f','%1.1f','%1.1f','%1.1f','%1.1f','%1.4e','%1.4e','%1.4e','%1.4e','%1.3f','%1.4e'], delimiter=delim)
    np.savetxt(prefix+datafile+"_parameters.txt", parameters_output, fmt="%s", delimiter=delim)
    
    # figure rendering
    
    if figure == True:
    
       figure_renderer(prefix, datafile, fig_suffix, lines[0], lines[1], lines[2], lines[3], chars[0], chars[1], chars[2])

        
    return
    
def figure_renderer(prefix, name, fig_suf, x_spec, y_spec, peak_line, baseline, peak_chars, ok_peaks, maybe_peaks):
    
    import math
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def normal_round(n, decimals = 0):
        
        expoN = n*10**decimals
        
        if abs(expoN) - abs(math.floor(expoN)) < 0.5:
            return math.floor(expoN)/10**decimals
        
        return math.ceil(expoN)/10**decimals
    
    canvas_size = (20, 10)
    labelsize = 20
    
    ok = peak_chars.T[2][ok_peaks]
    maybe = peak_chars.T[2][maybe_peaks]
    
    fig, ax = plt.subplots(figsize = canvas_size)

    ax.grid(which = "major")

    ax.plot(x_spec, y_spec, color = "black")
    ax.plot(x_spec, baseline, color = "C1")
    ax.plot(x_spec, peak_line, color = "C0")

    render = fig.canvas.get_renderer()

    text_heights = []

    for k in ok:
        t = ax.text(x_spec[k], y_spec[k], "  "+str(int(normal_round(x_spec[k]))), 
            color = "green", rotation = "vertical", ha = "center", va = "bottom", fontsize = labelsize)
        text_heights.append(t.get_window_extent(renderer = render).height)

    for k in maybe:
        t = ax.text(x_spec[k], y_spec[k], "  "+str(int(normal_round(x_spec[k]))), 
                color = "orange", rotation = "vertical", ha = "center", fontsize = labelsize)
        text_heights.append(t.get_window_extent(renderer = render).height)

    ax.tick_params(axis = "y", which = "major", direction = "in", right = True, 
                   length = 7, width = 2, labelsize = labelsize)


    ax.tick_params(axis = "x", which = "major", direction = "in", top = True, 
                   length = 7, width = 2, labelsize = labelsize)

    ax.set_xlim([np.amin(x_spec), np.amax(x_spec)])

    if len(text_heights) != 0:
        text_height = np.amax(text_heights)/1000
    else:
        text_height = 0

    y_lims = ax.get_ylim()
    y_range = y_lims[1] - y_lims[0]

    ax.set_ylim([y_lims[0], y_lims[1]+1.2*y_range*text_height])
    
    fig.savefig(prefix+name+fig_suf, bbox_inches = "tight")