"""
RamanMorph version 2.0.
1. run 'baseline_peakline_derivation' for setting the baseline and peakline.
2. run 'process_signals' for finding the peaks and computing heights and areas of them.
"""

import math
import numpy as np
from scipy import interpolate

class RamanMorph:
    def __init__(self, x, y, peak_hwindow, base_hwindow, limit_degrees=45, ok_degrees=30):
        """
        Initialize the RamanMorph object.
        :param x, y (array-like): The x/y-values of the Raman spectrum.
        :param peak_hwindow, base_hwindow: Half-window size for peakline/baseline detection, used in morphological operations.
        """
        self.x, self.y = x, y
        self.peak_hwindow, self.base_hwindow = peak_hwindow, base_hwindow
        self.limit_degrees, self.ok_degrees = limit_degrees, ok_degrees

        self.y_dilation = None
        self.y_erosion = None
        self.base_erosion = None

        self.tip_candidates = None
        self.tail_candidates = None
        self.base_tail_candidates = None

        self.peak_chars = [[], [], [], [], []]
        self.transposed_peak_chars = None

        self.edges = None
        self.slopes = None

        self.peakline = None
        self.baseline = None

        self.areas = None
        self.area_portions = None
        self.area_indices = None
        self.areas_sum = None

        self.peak_classification = None
        self.peak_order = None
        self.peak_normal_areas = None
        self.roc_derivative = None
        self.auc = None
        self.cumulative_area = None

        self.noise_std = None
        self.x_noise = None
        self.y_noise = None

        self.peak_chars_wavelengths = None
        self.peak_intensities = None
        self.peak_heights = None
        self.peak_baseline = None
        self.absolute_peak_heights = None
        self.angles = None

    def __str__(self):
        """
        String representation of the RamanMorph instance for print().
        """
        attributes = vars(self)
        return '\n'.join(f"{attribute}: {value}" for attribute, value in attributes.items())

    def __repr__(self):
        """
        Formal string representation of the RamanMorph instance.
        """
        attributes = vars(self)
        attributes_str = ', '.join(f"{attribute}={value!r}" for attribute, value in attributes.items())
        return f"{self.__class__.__name__}({attributes_str})"

    def _morphological_operation(self, half_window, operation):
        """
        A helper method for performing morphological operations (dilation or erosion).

        :param half_window (int) The half-window size for the operation.
        :param operation (callable): np.argmax for dilation or np.argmin for erosion.
        :return: Numpy array, the result of the morphological operation.
        """
        if not isinstance(half_window, int) or half_window <= 0:
            raise ValueError(f"'half_window' must be a positive integer: {half_window}")

        length = len(self.y)
        result_indices = np.zeros(length, dtype=int)

        for index in range(length):
            window_start = max(0, index - half_window)
            window_end = min(index + half_window + 1, length)
            window_range = np.arange(window_start, window_end)
            target_index = operation(self.y[window_range]) + window_start
            result_indices[index] = target_index

        return np.take(self.y, result_indices), np.where(result_indices - np.arange(length) == 0)[0]

    def dilation(self):
        """
        Perform dilation on the y-values of the Raman spectrum to identify peak candidates.
        """
        self.y_dilation, self.tip_candidates = self._morphological_operation(self.peak_hwindow, np.argmax)

    def erosion(self):
        """
        Perform two erosion operations on the y-values of the Raman spectrum to identify
        tail candidates of peakline and tail candidates of baseline.
        """
        self.y_erosion, self.tail_candidates = self._morphological_operation(self.peak_hwindow, np.argmin)
        self.base_erosion, self.base_tail_candidates = self._morphological_operation(self.base_hwindow, np.argmin)

    def _prepare_candidates(self):
        """
        Prepare the candidate lists by ensuring start and end points are included.
        """
        process_tails = lambda candidates: [0] + [tail for tail in candidates if tail not in [0, len(self.y) - 1]] + [len(self.y) -1]
        self.tail_candidates = process_tails(self.tail_candidates)
        self.base_tail_candidates = process_tails(self.base_tail_candidates)

    def _characterize_narrow_peaks(self):
        """
        Characterize narrow peaks.
        """
        for tip in self.tip_candidates:
            for i in range(1, len(self.tail_candidates)):
                if self.tail_candidates[i - 1] < tip < self.tail_candidates[i]:
                    self.peak_chars[1].append(self.tail_candidates[i - 1])
                    self.peak_chars[2].append(tip)
                    self.peak_chars[3].append(self.tail_candidates[i])
                    break

    def _characterize_wide_peaks(self):
        """
        Characterize wide peaks.
        """
        for tip in self.tip_candidates:
            for i in range(1, len(self.base_tail_candidates)):
                if self.base_tail_candidates[i - 1] < tip < self.base_tail_candidates[i]:
                    self.peak_chars[0].append(self.base_tail_candidates[i - 1])
                    self.peak_chars[4].append(self.base_tail_candidates[i])
                    break

    def _initial_peak_characterization(self):
        """
        Characterize initial peaks based on provided candidates.
        """
        self._characterize_narrow_peaks()
        self._characterize_wide_peaks()
        self.peak_chars = np.array(self.peak_chars).T

    def _find_minima_and_adjust_peaks(self):
        """
        Finds minima in multipeaks and adjust the peak characteriation.
        """
        new_borders = self.peak_chars[:, 1].copy()  # This is initially the left narrow edge

        for i in range(1, len(self.peak_chars)):
            if self.peak_chars[i, 1] == self.peak_chars[i - 1, 1]:
                # Calculate the difference array using slicing
                start_idx = self.peak_chars[i - 1, 2]  # This is the left peak tip
                end_idx = self.peak_chars[i, 2]  # This is the right peak tip
                diff = self.y.iloc[start_idx:end_idx] - self.y_erosion.iloc[start_idx:end_idx]

                # Find the index of the minimum difference and update new_borders
                new_border = np.argmin(diff) + start_idx
                self.peak_chars[i - 1, 3] = new_border
                new_borders[i] = new_border

        self.peak_chars[:, 1] = new_borders

    def _transpose_and_split(self):
        """
        Transpose and split peak characteristics for output.
        """
        self.transposed_peak_chars = self.peak_chars.T

    def peak_characterization(self):
        """
        Characterize peaks in the Raman spectrum.
        """
        self._initial_peak_characterization()
        self._find_minima_and_adjust_peaks()
        self._transpose_and_split()

    def _find_edges_and_minima(self, y_nominal, y_predef):
        """
        Find the edges and minima indices for interpolation.

        :param y_nominal (array-like): The original y-values.
        :param y_predef (array-like): The predefined y-values for interpolation.
        :return: List of lists, indices of right edges, minima and left edges.
        """
        im_pos = [[], [], []]  # [right_edge, minima_index, left_edge]

        for k in range(1, len(y_nominal)):
            if y_predef[k] > y_nominal[k] and y_predef[k - 1] <= y_nominal[k - 1]:
                im_pos[0].append(k)  # right edge
                for m in range(k, len(y_predef)):
                    if y_predef[m] > y_nominal[m]:
                        im_pos_end = m
                    else:
                        break
                im_pos[2].append(im_pos_end)  # left edge
            elif y_predef[k] == y_nominal[k]:
                im_pos[0].append(k)
                im_pos[2].append(k)

        for i, (start, end) in enumerate(zip(im_pos[0], im_pos[2])):
            if start != end:
                under_vals = y_nominal[start:end] - y_predef[start:end]
                min_ind = np.argmin(under_vals) + start
                im_pos[1].append(min_ind)
            else:
                im_pos[1].append(start)

        return im_pos

    def _linear_interpolate(self, y_nominal, im_pos):
        """
        Perform linear interpolation between identified points.

        :param y_nominal (array-like): The original y-values.
        :param im_pos (list of lists): Indices of right edges, minima, and left edges.
        :return (array-like): The interpolated y-values.
        """
        y_better = []
        prev_x, prev_y, prev_index = self.x[0], y_nominal[0], 0

        for k in im_pos[1]:
            dx = self.x[k] - prev_x
            dy = y_nominal[k] - prev_y
            if dy == 0:
                y_better.extend([prev_y] * (k - prev_index))
            else:
                slope = dy / dx
                y_better.extend([slope * (self.x[m] - prev_x) + prev_y for m in range(prev_index, k)])

            prev_x, prev_y, prev_index = self.x[k], y_nominal[k], k

        # Handle the final segment after the last minima to the end of the series
        if prev_index < len(y_nominal):
            y_better.extend([y_nominal[-1]] * (len(y_nominal) - prev_index))

        return y_better

    def _check_and_update_control(self, y_nominal, y_better):
        """
        Check if interpolation needs further iteration.

        :param y_nominal (array-like): The original y-values.
        :param y_better: The current interpolated y-values.
        :return (tuple: (bool, array-like)): Control flag and updated y-values.
        """
        control = True
        for k, val in enumerate(y_better):
            if val > y_nominal[k]:
                control = False
                break
        return control, y_better if control else y_better.copy()

    def improved_linear_interpolation(self, y_nominal, y_predefined, iteration_max=20):
        """
        Perform improved linear interpolation on the peakline.

        :param y_nominal (array-like): The original y-values.
        :param y_predefined (array-like): The predefined y-values for interpolation.
        :param iteration_max (int): Maximum number of iterations for the interpolation process.
        :return (array-like): Interpolated y-values.
        """
        iteration = 0
        control = False
        y_nominal, y_predefined = np.array(y_nominal), np.array(y_predefined)

        while not control and iteration <= iteration_max:
            im_pos = self._find_edges_and_minima(y_nominal, y_predefined)
            y_better = self._linear_interpolate(y_nominal, im_pos)
            control, y_predefined = self._check_and_update_control(y_nominal, y_better)
            iteration += 1

        return y_better

    def _determine_peak_edges(self):
        """
        Determine the edges of the peaks.
        """
        self.edges = [self.peak_chars[0][1]]

        if len(self.peak_chars) > 1:
            for k in range(1, len(self.peak_chars)):
                if self.peak_chars[k][1] != self.edges[-1]:
                    self.edges.append(self.peak_chars[k][1])
                if self.peak_chars[k][3] != self.edges[-1]:
                    self.edges.append(self.peak_chars[k][3])
        else:
            self.edges.append(self.peak_chars[0][3])  # If there is just one signal

        self.edges = list(set(self.edges))
        self.edges.sort()

    def _calculate_slopes(self):
        """
        Calculate the slopes between peak edges.
        """
        self.slopes = []
        for k in range(1, len(self.edges)):
            dy = self.y[self.edges[k]] - self.y[self.edges[k - 1]]
            dx = self.x[self.edges[k]] - self.x[self.edges[k - 1]]
            self.slopes.append(dy / dx)

    def _construct_peak_line(self):
        """
        Construct the peak line based on edges and slopes.
        """
        self.peakline = []
        c = 0
        for k in range(len(self.x)):
            if k < self.edges[0] or k > self.edges[-1]:
                self.peakline.append(self.y_erosion.iloc[k])
            elif self.edges[c] <= k < self.edges[c + 1]:
                self.peakline.append((self.x[k] - self.x[self.edges[c]]) * self.slopes[c] + self.y[self.edges[c]])
            elif k == self.edges[c + 1]:
                self.peakline.append(self.y[k])
                c = c + 1 if c + 1 != len(self.slopes) else c
            else:
                print("peakline: Something is wrong!")

    def peakline_derivation(self, iteration_max=20):
        """
        Drive the peakline from the characterized peaks.

        :param iteration_max: Maximum number of iterations for linear interpolation.
        """
        self._determine_peak_edges()
        self._calculate_slopes()
        self._construct_peak_line()
        self.peakline = self.improved_linear_interpolation(self.y, self.peakline, iteration_max=iteration_max)

    def _adjust_baseline_tails(self, peak):
        """
        Adjust the baseline tails of a single peak.

        :param peak (list): A list representing a single peak's characteristics.
        """
        # Adjust the left baseline tail
        for m in range(peak[0], peak[2]):
            if self.y[m] == self.baseline[m] and self.y[m] == self.peakline[m]:
                peak[0] = m

        # Adjust the right baseline tail
        for m in range(peak[2], peak[4]):
            if self.y[m] == self.baseline[m] and self.y[m] == self.peakline[m]:
                peak[4] = m
                break

    def _adjust_peakline_tails(self, peak):
        """
        Adjust the peakline tails of a single peak.

        :param peak (list): A list representing a single peak's characteristics.
        """
        # Adjust the left peakline tail
        for m in range(peak[1], peak[2]):
            if self.y[m] == self.peakline[m]:
                peak[1] = m

        # Adjust the right peakline tail
        for m in range(peak[2], peak[3]):
            if self.y[m] == self.peakline[m]:
                peak[3] = m
                break

    def _adjust_peak_tips(self, peak):
        """
        Adjust the peak tips to the maximum y value between the peak tails.

        :param peak (list): A list representing a single peak's characteristics.
        """
        start_idx, end_idx = peak[1], peak[3]
        peak_region = self.y.iloc[start_idx:end_idx+1] - self.peakline[start_idx:end_idx+1]
        peak_tip_idx = np.argmax(peak_region.values) + start_idx

        peak[2] = peak_tip_idx

    def tail_tip_correction(self):
        """
        Adjust the tails of peaks in 'self.peak_chars' based on baseline and peakline conditions.
        """
        for peak in self.peak_chars:
            self._adjust_baseline_tails(peak)
            self._adjust_peakline_tails(peak)
            self._adjust_peak_tips(peak)

        self._transpose_and_split()

    def _compute_narrow_and_wide_areas(self):
        """
        Compute the narrow and wide areas for each peak.
        :return: tuple of lists: (narrow_areas, wide_areas)
                    - narrow_areas: List of areas calculated between the peak line and the y-values for narrow peaks.
                    - wide_areas: List of areas calculated between the baseline and the y-values for wide peaks.
        """
        narrow_areas = [
            np.sum(
                self.y[self.peak_chars[k][1]:self.peak_chars[k][3] + 1] -
                self.peakline[self.peak_chars[k][1]:self.peak_chars[k][3] + 1]
            ) for k in range(len(self.peak_chars))
        ]

        wide_areas = []
        for k in range(len(self.peak_chars)):
            start_index = self.peak_chars[k][0]
            end_index = self.peak_chars[k][4] + 1
            is_new_peak = k == 0 or (self.peak_chars[k][0] != self.peak_chars[k-1][0] and self.peak_chars[k][4] != self.peak_chars[k-1][4])

            if is_new_peak:
                wide_area = np.sum(self.y[start_index:end_index] - self.baseline[start_index:end_index])
            else:
                wide_area = wide_areas[k-1]

            wide_areas.append(wide_area)

        return narrow_areas, wide_areas

    def _compute_total_areas(self, narrow_areas, wide_areas):
        """
        Compute the total area for each peak, adjusting for overlapping areas.

        :param narrow_areas (list): List of areas calculated between the peakline and the y-values for narrow peaks.
        :param wide_areas (list): List of areas calculated between the baseline and the y-values for wide peaks.
        :return: List of adjusted total areas for each peak.
        """
        areas = []
        for k, wide_area in enumerate(wide_areas):
            area_indices = [i for i, wa in enumerate(wide_areas) if wa == wide_area]
            if len(area_indices) == 1:
                total_area = narrow_areas[k]
            else:
                total_area = np.sum(np.array(narrow_areas)[area_indices])

            areas.append((narrow_areas[k] / total_area) * wide_area if total_area != 0 else 0)

        return areas

    def compute_areas(self):
        """
        Compute the areas under the peaks defined by peak characteristics.
        """
        narrow_areas, wide_areas = self._compute_narrow_and_wide_areas()
        self.areas = self._compute_total_areas(narrow_areas, wide_areas)
        self.areas_sum = np.sum(self.areas)
        self.area_portions = self.areas / self.areas_sum if self.areas_sum != 0 else self.areas
        self.area_indices = np.flip(np.argsort(self.areas), axis=0)

    def _calculate_peak_order_and_normal_areas(self, sorted_areas):
        """
        Calculate the peak order and normalized areas.

        :param sorted_areas:
        """
        self.peak_order = [float(k) / (len(self.peak_chars) - 1) for k in range(len(self.peak_chars))]
        self.peak_normal_areas = np.cumsum(sorted_areas) / self.areas_sum

        self.auc = np.sum(self.peak_normal_areas) / len(self.peak_order)

    def _interpolate_roc_curve(self):
        """
        Interpolate the ROC curve.
        """
        if len(self.peak_order) > 3:
            tck = interpolate.splrep(self.peak_order, self.peak_normal_areas, s=0)
        elif len(self.peak_order) == 3:
            tck = interpolate.splrep(self.peak_order, self.peak_normal_areas, s=0, k=2)
        elif len(self.peak_order) == 2:
            tck = interpolate.splrep(self.peak_order, self.peak_normal_areas, s=0, k=1)
        else:
            raise ValueError("Insufficient data for ROC analysis.")

        return tck

    def _calculate_roc_derivative(self, tck):
        """
        Calculate the derivative of the ROC curve.
        :param tck:
        :return:
        """
        return interpolate.splev(self.peak_order, interpolate.splder(tck))

    def _classify_peaks(self):
        """
        Classify peaks based on the derivative of the interpolated ROC curve.
        """
        tck = self._interpolate_roc_curve()
        self.roc_derivative = self._calculate_roc_derivative(tck)

        limit_tangent = np.tan(np.radians(self.limit_degrees))
        ok_tangent = np.tan(np.radians(self.ok_degrees))

        self.peak_classification = {'ok_peaks': [], 'limit_peaks': [], 'no_peaks': []}

        for k, derivative in enumerate(self.roc_derivative):
            if derivative >= limit_tangent:
                self.peak_classification['ok_peaks'].append(self.area_indices[k])
            elif limit_tangent > derivative >= ok_tangent:
                self.peak_classification['limit_peaks'].append(self.area_indices[k])
            elif derivative < ok_tangent:
                self.peak_classification['no_peaks'].append(self.area_indices[k])

    def perform_roc_test(self):
        """
        Perform ROC test to classify peak based on area and order.
        """
        self.areas = np.array(self.areas)
        sorted_areas = self.areas[self.area_indices]
        sorted_area_portions = self.area_portions[self.area_indices]
        self.cumulative_area = np.cumsum(sorted_area_portions)

        self._calculate_peak_order_and_normal_areas(sorted_areas)
        self._classify_peaks()

    def _get_valid_peaks(self):
        """
        Extract peaks that are classified as 'ok' or 'maybe'.
        :return: List of valid peak characters.
        """
        ok_peaks = self.peak_classification['ok_peaks']
        limit_peaks = self.peak_classification['limit_peaks']
        return [peak for peak in self.peak_chars if peak in self.peak_chars[ok_peaks] or peak in self.peak_chars[limit_peaks]]

    def _define_noise_regions(self, valid_peaks):
        """
        Define regions in the spectrum that are considered noise.

        :param valid_peaks: Indices of regions, where is a signal.
        :return: List of noise region start and end indices.
        """
        noise_regions = []
        for i, peak in enumerate(valid_peaks):
            if np.array_equal(peak, valid_peaks[0]) and peak[0] != 0:
                noise_regions.append([0, peak[0]])
            elif np.array_equal(peak, valid_peaks[-1]) and peak[-1] != len(self.y):
                noise_regions.append([valid_peaks[i - 1][2], peak[0]])
                noise_regions.append([peak[-1], len(self.y) - 1])
            else:
                noise_regions.append([valid_peaks[i - 1][2], peak[0]])

        return [region for region in noise_regions if region[0] != region[1]]

    def _extract_noise_data(self, noise_regions):
        """
        Extract x and y values for each noise region.

        :param noise_regions: Regions with noise.
        :return: Tuple of lists containing x and y coordinates of noise regions.
        """
        self.x_noise, self.y_noise = [], []
        for region in noise_regions:
            region_indices = np.arange(region[0], region[1])
            self.x_noise.append(self.x.iloc[region_indices])
            region_y_values = self.y.iloc[region_indices]
            self.y_noise.append(region_y_values)

    def compute_noise_deviation(self):
        """
        Compute the standard deviation of noise in the spectrum,
        excluding regions with 'ok' and 'limit' peaks.
        """
        valid_peaks = self._get_valid_peaks()
        noise_regions = self._define_noise_regions(valid_peaks)
        self._extract_noise_data(noise_regions)

        if self.y_noise:
            vals = np.concatenate(self.y_noise)
            self.noise_std = np.std(vals)
        else:
            print("Couldn't get enough data for computing standard deviation.")

    def _prepare_peak_output(self):
        """
        Prepare the peak output data for analysis and visualization.
        :return:
        """
        self.peakline, self.baseline = np.array(self.peakline), np.array(self.baseline)

        self.peak_chars_wavelengths = [self.x.iloc[char] for char in self.transposed_peak_chars[2]]
        self.peak_intensities = self.areas[self.area_indices]
        peak_indices = np.array(self.peak_chars).T[2][self.area_indices]
        self.absolute_peak_heights = self.y.iloc[peak_indices]
        self.peak_heights = self.y.iloc[peak_indices] - self.peakline[peak_indices]
        self.peak_baseline = self.y.iloc[peak_indices] - self.baseline[peak_indices]

        self.angles = [math.degrees(np.arctan(k)) for k in self.roc_derivative]


    def process_data(self):
        """
        Process the spectral data to compute peak areas, classify peaks, and calculate statistics.
        :return:
        """
        # Compute intensities (areas) of found peaks
        self.compute_areas()

        # Classify peaks
        self.perform_roc_test()

        # Extract noise-related parameters
        self.compute_noise_deviation()

        # Sort and prepare data for possible output
        self._prepare_peak_output()

    def baseline_peakline_derivation(self):
        """
        Derives the peakline and baseline.
        """
        self.dilation()
        self.erosion()
        self.peak_characterization()
        self.peakline_derivation()

        # Adjusts the baseline to be always below the spectrum
        self.baseline = self.improved_linear_interpolation(self.y, self.base_erosion)

        # Adjusts the baseline to be always below the peakline
        self.baseline = self.improved_linear_interpolation(self.peakline, self.baseline)

        self.tail_tip_correction()

    def process_signals(self):
        """
        Computes the signal parameters.
        """
        self.compute_areas()
        self.perform_roc_test()
        self.process_data()
