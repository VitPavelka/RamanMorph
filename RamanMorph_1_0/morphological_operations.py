# Definition of morphological oprators, with mirror boundary

def erosion(y_data_in, half_window):
    import numpy as np

    eros = np.zeros(len(y_data_in),dtype=int)
    for alfa in range(0,len(y_data_in)):
        start = max(0,alfa-half_window)
        end = min(alfa+half_window+1,len(y_data_in))
        pomo = np.arange( start,end )
        eros[alfa] = np.argmin(y_data_in[pomo])+start   # looking for minima


    y_eroded = np.take(y_data_in,eros) 
    y_contact = np.where( eros-np.arange(len(eros)) == 0 )[0]   #contact points
    return y_eroded,y_contact

def dilation(y_data_in, half_window):
    import numpy as np

    dila = np.zeros(len(y_data_in),dtype=int)
    for alfa in range(0,len(y_data_in)):
        start = max(0,alfa-half_window)
        end = min(alfa+half_window+1,len(y_data_in))
        pomo = np.arange( start,end )
        dila[alfa] = np.argmax(y_data_in[pomo])+start   # looking for maxima

    y_dilated = np.take(y_data_in,dila)
    y_contact = np.where( dila-np.arange(len(dila)) == 0 )[0]   #contact points
    return y_dilated,y_contact

def improved_linear_interpolation(x_spec, y_nominal, y_predef, iteration_max = 20):
    
    import numpy as np
    
    control = 0
    iteration = 0
    
    while control == 0 and iteration <= iteration_max:

        im_pos = [[],[],[]]  # list for edges and minima indices of signals [[right_edge], [index], [left_edge]]
        
        # FINDING EDGES
        for k in range(1, len(y_nominal)):   # looking forth and back
            if y_predef[k] > y_nominal[k] and y_predef[k - 1] <= y_nominal[k - 1]: # point has to be above 
                                                                                   # nominal y and previous 
                im_pos[0].append(k)  # right edge                                  # either under or on it
                
                for m in range(k, len(y_predef)):  # left edge
                    if y_predef[m] > y_nominal[m]:
                        im_pos_end = m
                    else:                            # breaks loop after finding left side
                        break

                k = m
                im_pos_end = k
                im_pos[2].append(im_pos_end)

            elif y_predef[k] == y_nominal[k]:  # this is for mutual edges of two signals
                im_pos[0].append(k)
                im_pos[2].append(k)
    
        # FINDING MINIMA
        for k in range(len(im_pos[0])):
            if im_pos[0][k] != im_pos[2][k]:  
                
                under_vals = [] # help list for finding minumum index
                
                for m in range(im_pos[0][k], im_pos[2][k]):
                    under_vals.append(y_nominal[m] - y_predef[m])
                
                min_ind = np.argmin(under_vals) + im_pos[0][k]  # finding index
                im_pos[1].append(min_ind)
            else:
                im_pos[1].append(im_pos[0][k]) # multipeaks have same minimum
                
            
        
        # LINEAR INTERPOLATION OF FOUND POINTS
        y_better = [] # final output  

        x = x_spec[0]
        y = y_nominal[0]
        index = 0
        
        for k in range(len(im_pos)):        
            im_pos[k].append(len(x_spec) - 1)
            
        for k in (im_pos[1]):

            dx = x_spec[k] - x
            dy = y_nominal[k] - y

            if dy == 0:  # if constant, y remains constant
                for m in range(index, k):
                    y_better.append(y_nominal[index])
            else:
                slope = dy/dx  # else computes slope

                for m in range(index, k):  # and then y values for straight lines between points
                    value = slope*(x_spec[m] - x_spec[index]) + y_nominal[index]
                    y_better.append(value)
                    
            x = x_spec[k]
            y = y_nominal[k]
            index = k
        y_better.append(np.array(y_nominal)[-1])  # this one is missing, key error, therefore flip
        
        control = 1  # tries to shut off the engine
        
        # ENGINE ISSUES
        for k in range(len(y_better)):  # if there are still areas under nominal y, iteration is needed
            if y_better[k] > y_nominal[k]:
                
                y_predef = y_better
                control = 0  # and sometimes it shut it on again
                iteration += 1
                
                break
                
    return y_better