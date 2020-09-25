import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import pyautogui
import datetime

# CONSi(L)
# Gregor Wirnsberger 07/2020

np.set_printoptions(threshold=sys.maxsize)

'CONCENTRATION CALCULATION ADVANCED SUM'


def get_resolution(pic):
    """
    gets the shape of the image and the resolution of the used screen to resize the image according to the screen
    INPUT:
        used picture
    OUTPUT:
        resolution to resize the image according to the screen
    """
    # get shape of the image (pixel x pixel)
    pc = cv2.imread(pic, 0)
    # getting the values to resize the image according to the display size
    res = (
        int(pc.shape[1] * (pyautogui.size()[0] / pc.shape[1])), int(pc.shape[0] * (pyautogui.size()[1] / pc.shape[0])))
    return res


def draw_rect(event, x, y, flags, param):
    """
    Draws a rectangle when mouse is clicked, moved and released
    INPUT:
        moving event and the x and y coordinates where rectangle is
    OUTPUT:
        a drawn rectangle on the global image
    """
    # mouse callback function
    global ix, iy, drawing, mode, ix_2, iy_2, img_draw_on
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        """
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.line(img_draw_on, (ix, iy), (x, y), (0, 255, 0), )
        """
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ix_2, iy_2 = x, y
        if mode:
            cv2.rectangle(img_draw_on, (ix, iy), (x, y), (0, 255, 0), )


def get_coord(recent_stat=None, offset=1):
    """
    gets the coordinates of the drawn rectangle in draw_rect
    INPUT:
        recent_stat: text that should be displayed on the image
        offset: offset from the left side of the image
    OUTPUT:
        Coordinates from the drawn rectangle
    """
    # binding mouse callback function to a window
    cv2.namedWindow('window')
    cv2.setMouseCallback('window', draw_rect)
    font = cv2.FONT_HERSHEY_COMPLEX
    while 1:
        cv2.imshow('window', img_draw_on)
        # to write text on image to tell you eg which step was just finished
        if recent_stat:
            cv2.putText(img_draw_on, recent_stat, (10 + int(offset), 20), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    # get right order of coordinates in coord_arr no matter where the drawing of the rectangle starts
    if iy < iy_2 and ix < ix_2:
        coord_arr = np.asarray([iy, iy_2, ix, ix_2])
    elif iy > iy_2 and ix < ix_2:
        coord_arr = np.asarray([iy_2, iy, ix, ix_2])
    elif iy < iy_2 and ix > ix_2:
        coord_arr = np.asarray([iy, iy_2, ix_2, ix])
    else:
        coord_arr = np.asarray([iy_2, iy, ix_2, ix])
    return coord_arr


def background_selection(bg_num=2):
    """
    loop for get_coord which is used to select as many backgrounds as you want and returns them as a list
    INPUT:
        How often the loop should be repeated- how many backgrounds should be selected in the image
    OUTPUT:
        list of background coordinates
    """
    bg_list = []
    # select as many background as bg_num
    for i in range(bg_num):
        bg_list += [get_coord()]
    cv2.imshow('window', img_draw_on)
    cv2.destroyAllWindows()
    return bg_list


# doesn't work as it should at the moment
def get_cor_fact(img_g_c_f, which_standard, base_peak_g_c_f):
    """
    not_enough = True
    coord_list = []
    while not_enough:
        coord_list += [get_coord().tolist()]
        if input("if all bands are selected press 'X' ").strip().upper() == "X":
            not_enough = False

    band_ind_list = input("Band indices": ).strip().split(",")
    """
    ###
    coord_list = [328, 365, 57, 157], [379, 416, 60, 156], [455, 494, 62, 159], [520, 561, 68, 158]

    band_ind_list = [7, 8, 9, 10]
    ###

    standard_conc = np.asarray(which_standard)[:, 1]

    band_ind_list = np.asarray(band_ind_list)
    used_conc = standard_conc[band_ind_list]
    conc_factors = used_conc.reshape(len(used_conc), 1) / used_conc

    h_list = []
    for i in coord_list:
        im = img_g_c_f[i[0]:i[1], i[2]:i[3]]
        h = np.histogram(im, 256, [0, 256])[0]
        h[:103] = 0
        h_list += [h]

    cor_list = []
    for i in range(2, 1237):
        cor_list += [np.arange(1, i, (i - 1) / 256).tolist()]

    st_lane_0_cor_300 = np.sum(h_list[0] * cor_list, axis=1)
    st_lane_1_cor_300 = np.sum(h_list[1] * cor_list, axis=1)
    st_lane_2_cor_300 = np.sum(h_list[2] * cor_list, axis=1)
    st_lane_3_cor_300 = np.sum(h_list[3] * cor_list, axis=1)

    lane_arr = np.asarray([st_lane_0_cor_300, st_lane_1_cor_300, st_lane_2_cor_300, st_lane_3_cor_300])
    hor = lane_arr.reshape(len(cor_list), len(coord_list), 1)
    ver = lane_arr.reshape(len(cor_list), 1, len(coord_list))
    cor_arr = (hor / ver) - conc_factors  # [:, :, np.newaxis]
    # print(np.abs(cor_arr))
    print(np.sum(np.sum(np.abs((hor / ver) - conc_factors), axis=1), axis=1))
    print(np.argmin(np.sum(np.sum(np.abs((hor / ver) - conc_factors), axis=1), axis=1)))
    print(max(np.sum(np.sum(np.abs((hor / ver) - conc_factors), axis=1), axis=1)))
    plt.plot(np.sum(np.sum(np.abs((hor / ver) - conc_factors), axis=1), axis=1))
    plt.show()


def concentration(img_c, conc_bg, show_plot=False, show_clac=False, show_selected_band=False):
    """
    calculates the difference in intensity by summing up all values (0-255) from the histogram of the two bands that
    should be compared against each other
    INPUT:
        img_c: the image containing the bands you want to compare
        conc_bg: list of the coordinates corresponding to the backgrounds selected before
        show_plot: if True- shows all histograms from the different calculation for the selected bands
        show_selected_band: if True shows the selected bands after all were selected to check if they are right
    OUTPUT:
        dictionary of all calculations (all different methods) with their sum of the bands and the factor how much more
        or less intense one compared to the other one is
    """
    # to get the coordinates for bands to compare
    cord_list = get_coord("background selection done"), get_coord()
    # get each selected band as own variable
    img_band_1 = img_c[cord_list[0][0]:cord_list[0][1], cord_list[0][2]:cord_list[0][3]]
    img_band_2 = img_c[cord_list[1][0]:cord_list[1][1], cord_list[1][2]:cord_list[1][3]]

    # list of all histograms of all selected backgrounds
    hist_base_list = []
    for i in range(len(conc_bg)):
        hist_base, bins_base = np.histogram(img_c[conc_bg[i][0]:conc_bg[i][1], conc_bg[i][2]:conc_bg[i][3]].ravel(),
                                            256, [0, 256])
        hist_base_list += [hist_base]
    hist_base_list = np.asarray(hist_base_list)
    # sum of all background histograms
    hist_base = (np.sum(hist_base_list, axis=0))

    # creates histograms from all selected bands
    hist_l_1, bins_l_1 = np.histogram(img_band_1.ravel(), 256, [0, 256])
    hist_l_2, bins_l_2 = np.histogram(img_band_2.ravel(), 256, [0, 256])

    # index of max value in base peak
    base_max_ind = np.argmax(hist_base)
    # most right peak in base peak
    peaks, _ = signal.find_peaks(hist_base, height=250)
    base_peak_rightest_peak = max(peaks)
    # makes a 1D array of it so signal.peak_widths can use it
    base_max_ind_1_d = np.asarray([np.argmax(hist_base)])
    # to get the most right point of the base peak
    base_peak_right = int(np.max(signal.peak_widths(hist_base, base_max_ind_1_d, 1)))

    'different correction factors to account for the reason that brighter pixel are more important'
    # linear factor- starts at 0 with 0 and end linear with 255 at 255
    cor_fact_256 = np.arange(1, 256, 255 / 256)
    """
    # linear factor- starts at most right point of base peak with 0 and end linear with x at 255
    cf_zeros = np.zeros(base_peak_right)
    cf_fact = np.arange(1, 100, (99 / (256 - base_peak_right)))
    print(len(cf_zeros), len(cf_fact))
    cor_fact_spec = np.concatenate((cf_zeros, cf_fact), axis=0)
    """
    cor_fact_spec = np.log2(np.arange(1, 257) * 32)

    # subtracts area of base peak and sets all negative numbers to 0
    hist_l_1_base_cor = hist_l_1 - hist_base
    hist_l_1_base_cor[hist_l_1_base_cor < 0] = 0
    hist_l_1_base_cor[:base_max_ind] = 0  # everything smaller than the max of base peak is set to 0 to get rid of bg
    hist_l_2_base_cor = hist_l_2 - hist_base
    hist_l_2_base_cor[hist_l_2_base_cor < 0] = 0
    hist_l_2_base_cor[:base_max_ind] = 0  # everything smaller than the max of base peak is set to 0 to get rid of bg
    # uses the base peak corrected histogram and uses  different correction factors
    double_cor_256_1 = hist_l_1_base_cor * cor_fact_256
    double_cor_spec_1 = hist_l_1_base_cor * cor_fact_spec
    double_cor_256_2 = hist_l_2_base_cor * cor_fact_256
    double_cor_spec_2 = hist_l_2_base_cor * cor_fact_spec

    # get sum of the histograms
    auc_base = np.sum(hist_base)
    # not corrected
    auc_nc_band1 = np.sum(hist_l_1)
    auc_nc_band2 = np.sum(hist_l_2)
    conc_nc = auc_nc_band1 / auc_nc_band2

    # highest point base peak
    auc_hpbp_band1 = np.sum(hist_l_1[base_max_ind:])
    auc_hpbp_band2 = np.sum(hist_l_2[base_max_ind:])
    conc_hpbp = auc_hpbp_band1 / auc_hpbp_band2

    # most right peak in base peak
    auc_rbp_band1 = np.sum(hist_l_1[base_peak_rightest_peak:])
    auc_rbp_band2 = np.sum(hist_l_2[base_peak_rightest_peak:])
    conc_rbp = auc_rbp_band1 / auc_rbp_band2

    # most right point base peak
    auc_rpbp_band1 = np.sum(hist_l_1[base_peak_right:])
    auc_rpbp_band2 = np.sum(hist_l_2[base_peak_right:])
    conc_rpbp = auc_rpbp_band1 / auc_rpbp_band2

    # subtraction base peak
    auc_sub_bp_band1 = np.sum(hist_l_1_base_cor)
    auc_sub_bp_band2 = np.sum(hist_l_2_base_cor)
    conc_sub_bp = auc_sub_bp_band1 / auc_sub_bp_band2

    # subtraction base peak and cor fact 256
    auc_cor_fact256_band1 = np.sum(double_cor_256_1)
    auc_cor_fact256_band2 = np.sum(double_cor_256_2)
    conc_cor_fact256 = auc_cor_fact256_band1 / auc_cor_fact256_band2

    # subtraction base peak and cor fact spec
    auc_cor_fact_spec_band1 = np.sum(double_cor_spec_1)
    auc_cor_fact_spec_band2 = np.sum(double_cor_spec_2)
    conc_cor_fact_spec = auc_cor_fact_spec_band1 / auc_cor_fact_spec_band2

    # dict with all sums and calculation results
    area_arr = {"base": auc_base,
                "nc": [auc_nc_band1, auc_nc_band2, conc_nc],
                "hpbp": [auc_hpbp_band1, auc_hpbp_band2, conc_hpbp],
                "rbp": [auc_rbp_band1, auc_rbp_band2, conc_rbp],
                "rpbp": [auc_rpbp_band1, auc_rpbp_band2, conc_rpbp],
                "sub_bp": [auc_sub_bp_band1, auc_sub_bp_band2, conc_sub_bp],
                "256": [auc_cor_fact256_band1, auc_cor_fact256_band2, conc_cor_fact256],
                "spec": [auc_cor_fact_spec_band1, auc_cor_fact_spec_band2, conc_cor_fact_spec]}

    if show_selected_band:
        cv2.imshow("band1", img_band_1)
        cv2.imshow("band2", img_band_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if show_clac:
        print("not corrected")
        print(auc_base)
        print(auc_nc_band1)
        print(auc_nc_band2)
        print(conc_nc)
        print("---------")
        print("corrected with highest point in base peak")
        print(auc_hpbp_band1)
        print(auc_hpbp_band2)
        print(conc_hpbp)
        print("---------")
        print("most right peak in base peak")
        print(auc_rbp_band1)
        print(auc_rbp_band2)
        print(conc_rbp)
        print("---------")
        print("corrected with rightest point of base peak")
        print(auc_rpbp_band1)
        print(auc_rpbp_band2)
        print(conc_rpbp)
        print("---------")
        print("corrected by subtracting the base peak area")
        print(auc_sub_bp_band1)
        print(auc_sub_bp_band2)
        print(conc_sub_bp)
        print("---------")
        print("corrected by subtracting the base peak area and multiplying it with the cor_fact_256")
        print(auc_cor_fact256_band1)
        print(auc_cor_fact256_band2)
        print(conc_cor_fact256)
        print("---------")
        print("corrected by subtracting the base peak area and multiplying it with the cor_fact_sepc")
        print(auc_cor_fact_spec_band1)
        print(auc_cor_fact_spec_band2)
        print(conc_cor_fact_spec)

    if show_plot:
        c_1, = plt.plot(hist_l_1_base_cor, label="cor_1")
        c_2, = plt.plot(hist_l_2_base_cor, label="cor_2")
        l1, = plt.plot(hist_l_1, label="line_1")
        l2, = plt.plot(hist_l_2, label="line_2")
        b, = plt.plot(hist_base, label="base")
        plt.legend(handles=[b, c_1, c_2, l1, l2], loc="upper right")
        plt.show()

        dc_256_1, = plt.plot(double_cor_256_1, label="dcor256_1")
        dc_256_2, = plt.plot(double_cor_256_2, label="dcor256_2")
        dc_spec_1, = plt.plot(double_cor_spec_1, label="dcor_spec_1")
        dc_spec_2, = plt.plot(double_cor_spec_2, label="dcor_spec_2")
        b, = plt.plot(hist_base, label="base")
        plt.legend(handles=[b, dc_256_1, dc_256_2, dc_spec_1, dc_spec_2], loc="upper right")
        plt.show()

    return area_arr


def abs_concentration(concentration_factor):
    """
    calculation of the absolute concentration of the band which got compared to the standard band
    INPUT:
        factor how much more or less intense the band compared to the standard band is from 'concentration'
    OUTPUT:
        absolute concentration calculated by the the factor by using the concentration from the used standard
    """
    # prints the list of supported standards so you know which standard has which index
    supported_standards()
    u_std = int(input("Which standard did you use? please enter number from the list above\n"))
    u_lane = int(input("Which standard lane did you use to compare your lane to? please start counting with 0\n"))
    # gets the concentrations of the standard you used
    std = read_standard_file("supported_standards.txt", u_std)[1]
    conc_abs = std[u_lane] * concentration_factor
    return conc_abs


# ------------------------------------------------------------------------------------------------------------------- #
'STANDARD SEPARATION'


def rotation(img_part, s_mode):
    """
    rotates the image so it can get summed up to get peaks where the highest intensity is (where bands in the
    image area)
    INPUT:
        image that should be rotated (standard) - not only coordinates but cropped image
    OUTPUT:
        rotated image
    """
    s_mode = int(s_mode)
    # dimensions and center
    (h, w) = img_part.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    # get rotation matrix +90 -> 90 degree turn counterclockwise
    matrix = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    # new bounds
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    # adjusting rotation matrix
    matrix[0, 2] += (new_width / 2) - cx
    matrix[1, 2] += (new_height / 2) - cy
    img_rot = cv2.warpAffine(img_part, matrix, (new_width, new_height))

    # get rid of background
    s_mode_list = [50, 60, 80, 90, 100, 100, 100]
    img_rot[img_rot < s_mode_list[s_mode]] = 0

    # noise reduction
    img_rot = cv2.bilateralFilter(img_rot, 9, 75, 75)

    return img_rot


def lane_add(la_ori_img, la_draw_img=None, finish_text=None, la_offset=0):
    """
    to get the middle of a band where the highest intensity is
    INPUT:
        la_ori_img: image where the middle of the band gets calculated
        la_draw_img: image where the rectangles to select the band is gets drawn
        finish_text: if True a text that gets displayed on the image
        la_offset: how much away from the left side the text should be
    OUTPUT:
        mid: middle of the band but only the y coordinates from the top of the image
        mid_coord: coordinates of the selected band corresponding to the image
    """
    la_coord_arr = get_coord(finish_text, la_offset)
    # middle (where the maximum is) of the lane
    la_img_flat = np.sum(rotation(la_ori_img[la_coord_arr[0]:la_coord_arr[1], la_coord_arr[2]:la_coord_arr[3]], 1),
                         axis=0)
    # middle with added the distance from the top of the image
    mid = np.argmax(la_img_flat) + la_coord_arr[0]
    # shows middle of newly selected lane on the image
    la_marker_mid = ((la_draw_img[la_coord_arr[0]:la_coord_arr[1], la_coord_arr[2]:la_coord_arr[3]].shape[1] / 2) +
                     la_coord_arr[2])
    la_draw_img = cv2.circle(la_draw_img, (int(la_marker_mid), int(mid)), 5, (0, 0, 255), )
    cv2.imshow("window", la_draw_img)
    cv2.moveWindow("window", 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mid_coord = np.asarray([int(la_marker_mid), int(mid)])
    return mid, mid_coord


def std_lane_recognition(ori_img, draw_img, coord_arr, sensitivity_mode, show_lines=False, draw_lines=False,
                         show_plot=False):
    """
    gets where the bands in the selected standards are
    INPUT:
        ori_img: image used to calculate the middle of the bands in the selected standard
        draw_img: image where the selection are getting drawn on
        coord_arr: coordinates of the standard in the image
        sensitivity_mode: how sensitive the selection should work- for bands with low intensity 0 with high 6
        show_lines: if True shows bands that got recognized
        draw_lines: if True shows lines around the bands in the standard
        show_plot: if True shows plot of intensity calculated by rotating the picture and summing it up along axis=0
    OUTPUT:
        peaks_std: middle of the recognized bands according to the top of the selection of the standard
        lines: list of coordinates around the recognized images
        peaks_std_top: as peaks_std but from the top of the image
        std_length: length from the top of the standard to the bottom
    """
    sensitivity_mode = int(sensitivity_mode)
    # sums up along lanes
    img_flat = np.sum(rotation(ori_img[coord_arr[0]:coord_arr[1], coord_arr[2]:coord_arr[3]], sensitivity_mode), axis=0)
    # get peaks, their properties as well as their prominence (height over their baseline)
    peaks_std, peak_prop = signal.find_peaks(img_flat, height=4000, distance=10, width=2)
    peak_prom = signal.peak_prominences(img_flat, peaks_std)

    # get rid of peaks produced by noise or other none lane stuff -> peaks with a lower prominence get ignored
    peak_prom_list = [50, 70, 100, 200, 300, 650, 900]
    peaks_std = np.delete(peaks_std, np.where(peak_prom[0] < peak_prom_list[sensitivity_mode]))

    # for low intensity lanes
    # peaks_std, peak_prop = signal.find_peaks(img_flat, height=200, distance=10, width=2, threshold=1)

    left_base = np.asarray(peak_prop["left_bases"], dtype=int)
    right_base = np.asarray(peak_prop["right_bases"], dtype=int)
    l_r_dif = left_base - right_base
    # standard length
    std_length = right_base[-1] - left_base[0]

    # get rid of to big distances - when left_ips and right_ips are false for drawing lines around lanes
    to_big_dist = np.where(l_r_dif < 2 * np.median(l_r_dif))
    left_base = np.delete(left_base, to_big_dist)
    right_base = np.delete(right_base, to_big_dist)
    img_flat = np.append(img_flat, np.zeros(coord_arr[1] - len(img_flat)))

    # shows the rotated and summed up on axis 0 picture and the intensity peaks as well as which peaks are recognized
    if show_plot:
        plt.plot(img_flat)
        plt.plot(peaks_std, img_flat[peaks_std], "x")
        plt.plot(left_base, img_flat[left_base], "x")
        plt.plot(right_base, img_flat[right_base], "x")
        plt.figure()
        plt.imshow(rotation(ori_img[coord_arr[0]:coord_arr[1], coord_arr[2]:coord_arr[3]], sensitivity_mode))
        plt.show()

    # print circle where middle of band is
    marker_middle = (draw_img[coord_arr[0]:coord_arr[1], coord_arr[2]:coord_arr[3]].shape[1] / 2) + coord_arr[2]
    lines = []
    slr_count = 0
    line_count = 0
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in peaks_std:
        # circle in the middle of the lane (where peak is) and label with index on the shown image
        draw_img = cv2.circle(draw_img, (int(marker_middle), int(i) + coord_arr[0]), 5, (0, 0, 255), )
        draw_img = cv2.putText(draw_img, str(line_count), (int(marker_middle) + 30, int(i) + coord_arr[0]),
                               font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        line_count += 1
        if draw_lines:
            # draws lines around lanes
            draw_img = cv2.rectangle(draw_img, (coord_arr[3], right_base[slr_count] + coord_arr[0]),
                                     (coord_arr[2], left_base[slr_count] + coord_arr[0]), (255, 255, 255), )
        lines += [[left_base[slr_count] + coord_arr[0], right_base[slr_count]
                   + coord_arr[0], coord_arr[2], coord_arr[3]]]
        if slr_count < len(left_base) - len(to_big_dist):
            slr_count += 1
    # shows lines around lanes
    if show_lines:
        for i in lines:
            cv2.imshow(str(i), draw_img[i[0]:i[1], i[2]:i[3]])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cv2.imshow("window", draw_img)
    cv2.moveWindow("window", 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # whether the standard lane detection is right, needs more lanes or needs some lanes added
    q = input("|****************************************************************|\n"
              "|are all bands detected?                                         |\n"
              "|if yes enter 'y'                                                |\n"
              "|if to many are detected (false positive) enter 'm'              |\n"
              "|if bands are missing enter 'a'                                  |\n"
              "|if bands are missing and false positives are detected enter 'b' |\n"
              "|confirm with pressing enter                                     |\n"
              "|****************************************************************|\n").strip().lower()
    # delete excess lanes in peaks_std
    if q == "m":
        many = input("Which lanes are to much? (please enter the indices which were shown in the image)\n")
        peaks_std = np.delete(peaks_std, np.asarray(many.split(","), dtype=int))
    # add missing lanes to peaks_std
    elif q == "a":
        less = int(input("How many lanes are missing? (please enter an integer)\n"))
        m_l_to_add = []
        for i in range(less):
            m_l_to_add += [lane_add(img_draw, img_draw_on)[0] - coord_arr[0]]
        peaks_std = np.append(peaks_std, np.asarray(m_l_to_add))
        peaks_std = np.sort(peaks_std)
    # some lanes are false positive some are missing
    elif q == "b":
        many_a_l = input("Which lanes are to much? (please enter the indices which were shown in the image)\n")
        peaks_std = np.delete(peaks_std, np.asarray(many_a_l.split(","), dtype=int))
        m_a_less = int(input("How many lanes are missing? (please enter an integer)\n"))
        m_l_to_add = []
        for i in range(m_a_less):
            m_l_to_add += [lane_add(img_draw, img_draw_on)[0] - coord_arr[0]]
        peaks_std = np.append(peaks_std, np.asarray(m_l_to_add))
        peaks_std = np.sort(peaks_std)
    # all lanes are correct
    elif q == "y":
        peaks_std = peaks_std
    elif q != "y":
        print("wrong input")

    # middle of line of peaks measured from top of the image not from the top of the selection like in peaks_std
    peaks_std_top = peaks_std + coord_arr[0]

    # peaks_std = middle of line
    return peaks_std, lines, peaks_std_top, std_length


# ------------------------------------------------------------------------------------------------------------------- #
'SIZE CALCULATION'


def size_calculations(standard_peaks, standard_size_conc, bands_oi, plot_show=False):
    """
    calculates the size of the band of interest which is calculated corresponding to the migration distances in the
    standard
    INPUT:
        standard_peaks: migration distances of the standard in the image
        standard_size_conc: band sizes of the used standard
        bands_oi: migration distances from the bands of interest
        plot_show: if True shows the curve of migration distance against their size and the calculated sizes as 'x'
    OUTPUT:
        size of the bands of interests according to their migration length
    """
    # x coordinates and y coordinates
    standard_size_conc = np.asarray(standard_size_conc)
    standard_peaks = np.asarray(standard_peaks)
    # least square polynomial fit
    z = np.polyfit(standard_peaks, standard_size_conc, 9)
    # 1D polynomial class
    pol = np.poly1d(z)
    # calculated size with the obtained polynomial equation in pol
    size = pol(np.asarray(bands_oi))
    # shows where the calculated size of the band lies
    c_peaks = []
    for i in standard_peaks:
        c_peaks += [pol(i)]
    if plot_show:
        real, = plt.plot(standard_peaks, standard_size_conc, label="real")
        calc, = plt.plot(standard_peaks, c_peaks, label="calc")
        plt.legend(handles=[real, calc], loc="upper right")
        plt.plot(bands_oi, pol(bands_oi), "x")
        plt.show()
    return size


def size_lane_add(lane_num=1):
    """
    same as lane_add but to get lane_adds output as often as you want and get it as a list back and buts the indexes of
    the selected band beside the band which got selected
    INPUT:
        how often the loop should be repeated
    OUTPUT:
        list of lane_add[0]
        list of lane_add[1]
    """
    global img_draw_on
    size_lane_add_list = []
    lane_coord_list = []
    sla_count = 0
    for i in range(lane_num):
        lane_var = lane_add(img_draw, img_draw_on, "select bands of interest")
        size_lane_add_list += [lane_var[0]]
        lane_coord_list += [lane_var[1]]
        cv2.putText(img_draw_on, str(sla_count), (lane_var[1][0], lane_var[1][1] - 40),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        sla_count += 1
    return np.asarray(size_lane_add_list), np.asarray(lane_coord_list)


def read_standard_file(file, ind_u_std=0):
    """
    reads the file in which each standard with its corresponding size and concentration of its bands
    INPUT:
        file with size and concentrations in the supported standard
    OUTPUT:
        std_size:
            sizes of all bands in the used standard
        std_conc:
            concentration of all bands in the used standard
    """
    # get all supported standards
    std = open(file, "r")
    used_std = std.readlines()
    std.close()
    # splits the line so every entry in line is its own element
    line = np.asarray(used_std[ind_u_std].strip().split(","), dtype=float)
    # sizes and concentrations of the used standard
    std_sizes = line[np.arange(0, len(line), 2)]
    std_conc = line[np.arange(1, len(line), 2)]

    return std_sizes, std_conc


def supported_standards(ret=False):
    """
    prints all supported standards which are in the supported_standards.txt
    INPUT:
        ret: if True returns the list of supported standards with their name and concentration
    OUTPUT:
        list of supported standards with their name and concentration
    """
    standard_list = [["0 -- 1 kb DNA ladder Solis BioDyne", "ng/10µl"],
                     ["1 -- 100 bp DNA Ladder Ready to Load", "ng/10µl"],
                     ["2 -- 100 bp quick load ladder new england biolabs", "ng/µl"]]
    print("supported standards")
    for i in standard_list:
        print(i[0])
    if ret:
        return standard_list


# ------------------------------------------------------------------------------------------------------------------- #
'ACCESSING'


def image_list():
    """
    Whether the image you want to use is one of the images.txt list or a new one, with the ability to clear the list and
    the check whether a new image isn't already in the list to prevent it from creating double entries
    OUTPUT:
        the name of the chosen image as a string
    """
    # file which stores the recently opened images
    open_images = open("images.txt", "r")
    recent_img = open_images.readlines()
    open_images.close()
    # prints all recently opened images and their according indices so you don't have to type the filename all the time
    count = 0
    rec_img_list = []
    if len(recent_img) == 0:
        pass
    else:
        for i in range(len(recent_img)):
            rec_img_list += [recent_img[i].rstrip()]
            print(str(count) + " -- " + recent_img[i].rstrip())
            count += 1
    # name of the image you want to open
    img_name = input("Enter image name or if image of choice is in the list above enter its index or enter 'clear'"
                     " to delete the image list and press enter\n")
    # get name from images.txt
    if img_name.strip().lower().isdigit():
        chosen_img = rec_img_list[int(img_name.strip())]
    # to clear images.txt if the list gets to long
    elif img_name.strip().lower() == "clear":
        c_file = open("images.txt", "w")
        c_file.write("")
        c_file.close()
        chosen_img = input("Enter new image name\n").rstrip()
        n_file = open("images.txt", "w")
        n_file.write(chosen_img + "\n")
        n_file.close()
    # if image name is not in images.txt
    else:
        chosen_img = img_name.rstrip()
        # appending images.txt with the new image name, but only if this image is not already in there
        if img_name.strip() not in rec_img_list:
            append = open("images.txt", "w")
            append.write(chosen_img + "\n")
            append.write("".join(recent_img))
            append.close()

    return chosen_img


def check_coord(used_img):
    """
    show the image where the rectangles got selected with matplotlib to get coordinates with the mouse
    INPUT:
        used image
    OUTPUT:
        shows the image
    """
    plt.figure()
    plt.imshow(used_img)
    plt.show()


def what():
    """
    getting input about what should be executed
    """
    q_w = input("for concentration calculation enter 'conc'\n"
                "for size calculation enter 'size'\n").strip().lower()
    if q_w == "conc":
        print("|--------------------------------------------------------------------------------------------------|\n"
              "| please be aware- the closer the migration distance of your sample band is to the standard band   |\n"
              "| you want to compare it to, the more accurate the result will be                                  |\n"
              "| the more backgrounds you select and the smaller they are, the more accurate the result will be   |\n"
              "| Please confirm every selected standard with pressing 'esc'                                       |\n"
              "|--------------------------------------------------------------------------------------------------|\n")
        bg = int(input("Enter how many backgrounds you want to select\n"))
        print("Please switch windows and select the backgrounds now")
        c_fact = concentration(img_draw, background_selection(bg))["hpbp"][2]
        print(str(int(np.round(abs_concentration(c_fact)))) + "ng/10µl")
    elif q_w == "size":
        sen_mode = int(input("Enter sensitivity mode (0,1,2,3,4,5,6 with 0 is most 6 is least)\n"))
        supported_standards()
        size_u_std = int(input("Which standard did you use? please enter number from the list above.\n"))
        print("please select a standard lane in your image")
        standard_bands = std_lane_recognition(img_draw, img_draw_on, get_coord(), sen_mode)[2]
        standard_prop = read_standard_file("supported_standards.txt", size_u_std)[0]
        added_bands = size_lane_add(int(input("how many sizes do you want to calculate?\n")))
        calculated_sizes = np.asarray(np.round(size_calculations(standard_bands, standard_prop, added_bands[0])),
                                      dtype=int)
        for i, j in zip(added_bands[1], calculated_sizes):
            cv2.putText(img_draw_on, str(j) + "bp", (i[0] - 42, (i[1] + 40)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("window", img_draw_on)
        cv2.moveWindow("window", 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        band_size_print_count = 0
        for k in calculated_sizes:
            print("band " + str(band_size_print_count) + " -- " + str(k) + "bp")
            band_size_print_count += 1
    else:
        print("incorrect input")


def view_logo(over_time):
    """
    shows the CONSiL logo at the start
    INPUT:
        if time is up so logo gets shown after specific amount of time
    """
    if over_time:
        logo = cv2.imread("CONSi(L)_logo_black.PNG")
        l_res = get_resolution("CONSi(L)_logo_black.png")
        logo = cv2.resize(logo, l_res)
        cv2.putText(logo, "press any key to continue", (int(l_res[0] / 2) - 230, l_res[1] - 65),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("start", logo)
        cv2.moveWindow("start", 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def logo_time(time_pass):
    """
    whether it's time to show the logo at the start again
    INPUT:
        how much time in minutes should pass until the logo shows up again at the start
    OUTPUT:
        if True specified time has passed (so in view_logo the viewing of the logo gets activated)
    """
    logo_file = open("s_logo.csv", "r")
    logo_times = logo_file.readlines()
    logo_file.close()
    if (datetime.datetime.now() - datetime.datetime.strptime(logo_times[0].rstrip(), "%Y-%m-%d %H:%M:%S")) > \
            datetime.timedelta(minutes=int(time_pass)):
        logo_f = open("s_logo.csv", "w+")
        logo_f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logo_f.close()
        return True


#######################################################################################################################
#######################################################################################################################

view_logo(logo_time(15))
drawing = False  # true if mouse is pressed
mode = True  # if true, draw rectangle
ix, iy = -1, -1
ix_2, iy_2 = -1, -1
picture = image_list()
resolution = get_resolution(picture)
img = cv2.imread(picture, 0)
img_draw = cv2.resize(img, resolution)
img_draw_on = cv2.resize(img, resolution)
what()

#######################################################################################################################

'CONCENTRATION CALCULATION ADVANCED SUM'

# print(concentration(img_draw, background_selection(4))["hpbp"][2])

# start = timer()
# print(get_cor_fact(img_draw, standard_1kb_size_conc, 102))
# end = timer()
# print(end - start)
# ------------------------------------------------------------------------------------------------------------------- #
'STANDARD SEPARATION'
# print(std_lane_recognition(img_draw, img_draw_on, get_coord(), 5, show_plot=True)[2].tolist())
# print(lane_add(img_draw, img_draw_on))
# ------------------------------------------------------------------------------------------------------------------- #
'SIZE CALCULATION'
# print(size_calculations(std_lane_recognition(img_draw, img_draw_on, get_coord(), 6)[2],
#                         read_standard_file("supported_standards.txt", 0)[0],
#                        lane_add(img_draw, img_draw_on "select band of interest")[0], plot_show=True))
# ------------------------------------------------------------------------------------------------------------------- #
'ACCESSING'
# image_list()
# what()
# check_coord(img_draw)
# recent_img_log(picture)
#######################################################################################################################
#######################################################################################################################

