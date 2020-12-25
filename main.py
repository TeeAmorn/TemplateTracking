import cv2 as cv
import numpy as np
import os
from itertools import product

# =============== Matching Methods ===============

def ssd(can, tpl):
    """
    Implementation of the the sum of square difference matching method.
    """
    score = 1/(np.sum(np.power(can-tpl, 2)))
    return score

def cc(can, tpl):
    """
    Implementation of the cross-correlation matching method.
    """
    score = np.sum(np.multiply(can, tpl))
    return score

def ncc(can, tpl):
    """
    Implementation of the the normalized cross-correlation matching method.
    """
    can_mean = np.mean(can)
    tpl_mean = np.mean(tpl)
    numerator = np.sum(np.multiply(can-can_mean, tpl-tpl_mean))
    can_std = np.std(can)
    tpl_std = np.std(tpl)
    denominator = can_std * tpl_std
    score = numerator / denominator
    return score

# =============== Searching Method ===============

def search(search_window, template, template_center, match_method):
    """
    Implementation of the exhausive search method.

    Parameters
    ----------
    search_window : The portion of the input image to perform exhausive search
        on. Every possible box of size template.shape will be considered as
        a possible candidate.
    template : The tracked/matched image from the previous frame.
    template_center : The (y, x) coordinate of the template's center relative
        to the search window. The top-left corner of the search window has
        coordinate (0, 0). Note that template_center is passed as a tuple.
    match_method : The matching method to be used in the exhausive search.

    Returns
    -------
    new_template : The new tracked/matched image.
    relative_position : The change in the number of vertical and horizontal
        pixels from the previous image center. It is returned as a tuple.
    """

    # Obtain size of search window and template
    win_height, win_width = search_window.shape
    tpl_height, tpl_width = template.shape

    # Match the template to every possible candidate in the search window
    scores = np.zeros((win_height-tpl_height, win_width-tpl_width))
    for (y, x) in product(range(win_height-tpl_height), range(win_width-tpl_width)):
        candidate = search_window[y:(y+tpl_height), x:(x+tpl_width)]
        scores[y, x] = match_method(candidate, template)
    max_score_y, max_score_x = np.unravel_index(scores.argmax(), scores.shape)

    # Obtain new template
    new_template = search_window[max_score_y:(max_score_y+tpl_height),
                                 max_score_x:(max_score_x+tpl_width)]

    # Compute the change in number of pixels from previous center
    center_y = max_score_y + (tpl_height-1)//2
    center_x = max_score_x + (tpl_width-1)//2
    dy = center_y - template_center[0]
    dx = center_x - template_center[1]

    return new_template, (dy, dx)

# =============== Target Tracking Implementation ===============

def draw_box(img, pos, box_height, box_width):
    start_point = (pos[1] - box_width//2, pos[0] - box_height//2)
    end_point = (pos[1] + box_width//2 + 1, pos[0] + box_height//2 + 1)
    color = (255, 0, 0)
    thickness = 2
    tracked_image = cv.rectangle(img, start_point, end_point, color, thickness)
    return tracked_image

# =============== Target Tracking Implementation ===============

def track(images_dir, first_template, first_pos, match_method):
    """
    Implementation of the template-matching based target tracking algorithm.
    """
    isFirst = True
    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith(".jpg"):
            frame = cv.imread(os.path.join("video", filename), cv.IMREAD_GRAYSCALE)
            frame_colored = cv.imread(os.path.join("video", filename))
            frame_height, frame_width = frame.shape
            if (isFirst):
                tpl_height, tpl_width = first_template.shape
                prev_tpl = first_template
                curr_pos = first_pos
                isFirst = False
            else:
                prev_tpl, dydx = search(frame, prev_tpl, curr_pos, match_method)
                curr_pos = (curr_pos[0]+dydx[0], curr_pos[1]+dydx[1])
            tracked_frame = draw_box(frame_colored, curr_pos, 41, 41)
            cv.imwrite(os.path.join("main_cc_results_43", filename), tracked_frame)


# =============== Test ===============

img_path = os.path.join("video", "0001.jpg")
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

# tpl = img[34:63, 59:88] # 25x25
# track("video2", tpl, (48, 73), ncc) # 25x25

# tpl = img[20:67, 49:96] # 47x47
# track("video", tpl, (48, 73), ncc) # 47x47

tpl = img[22:65, 50:93] # 43x43
track("video", tpl, (43, 71), cc) # 43x43