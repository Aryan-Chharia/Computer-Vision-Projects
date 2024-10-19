import cv2
import imutils
import numpy as np

def resize_image(input_image, new_width):
    orig_width, orig_height = input_image.shape[1], input_image.shape[0]
    ratio = new_width / float(orig_width)
    new_height = int(orig_height * ratio)
    dim = (new_width, new_height)
    reshaped_image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    return reshaped_image


def to_grayscale(img, method="mean", blocksize=91, c=7):
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "mean":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(gray,
                                    maxValue=255,
                                    adaptiveMethod=adaptiveMethod,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=blocksize,
                                    C=c)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def get_ordered_corners(approx_arr):
    try:
        assert(approx_arr.shape == (4, 1, 2) or approx_arr.shape == (4, 2))
    except:
        raise ValueError(f"Incorrect shape for approx_arr: {approx_arr.shape}. Requires shape of (4, 1, 2) or (4, 2).")
    if approx_arr.shape == (4, 1, 2):
        approx_arr = np.squeeze(approx_arr, axis=1)
    max_x = int(1.1 * np.max(approx_arr[:,0]))
    origin_1 = [0, 0]
    origin_2 = [max_x, 0]
    distances_1 = [np.linalg.norm(point - origin_1) for point in approx_arr]
    distances_2 = [np.linalg.norm(point - origin_2) for point in approx_arr]
    tl_idx = np.argmin(distances_1)
    br_idx = np.argmax(distances_1)
    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = np.inf
    dist_arr[br_idx] = np.inf
    tr_idx = np.argmin(dist_arr)
    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = -np.inf
    dist_arr[br_idx] = -np.inf
    bl_idx = np.argmax(dist_arr)
    tl = approx_arr[tl_idx]
    br = approx_arr[br_idx]
    tr = approx_arr[tr_idx]
    bl = approx_arr[bl_idx]
    return np.array([tl, tr, br, bl])


def perspective_transform(input_img, src_corners, pad=10):
    src_corners = get_ordered_corners(src_corners)
    src_corners = src_corners.astype("float32")
    tl, tr, br, bl = src_corners
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(bottom_width), int(top_width))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))
    dest_img_corners = np.array([[0+pad, 0+pad],
                                 [max_width-1-pad, 0+pad],
                                 [max_width-1-pad, max_height-1-pad],
                                 [0+pad, max_height-1-pad]], dtype="float32")
    M = cv2.getPerspectiveTransform(src=src_corners, dst=dest_img_corners)
    warped_img = cv2.warpPerspective(input_img, M, (max_width, max_height))
    return M, warped_img


def find_grid_candidates(img, to_plot=False):
    M_matrices = []
    warped_images = []
    img_area = img.shape[0] * img.shape[1]
    thresh = to_grayscale(img, blocksize=41, c=8)
    contours = cv2.findContours(image=thresh.copy(),
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if to_plot:
            with_contours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 75), thickness=2)
            cv2.imshow("Contours", with_contours)
            cv2.waitKey(0)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            contour_area = cv2.contourArea(contour)
            contour_fractional_area = contour_area / img_area
            if len(approx) == 4 and contour_fractional_area > 0.1:
                approx = get_ordered_corners(approx)
                M, warped_img = perspective_transform(input_img=img,
                                                              src_corners=approx,
                                                              pad=30)
                M_matrices.append(M)
                warped_images.append(warped_img)
    if warped_images:
        return M_matrices, warped_images
    else:
        raise Exception("No grid contour candidates were found in image")


def detect_digit(img, area_threshold=5, apply_border=False):
    cell_img = img.copy()
    if apply_border:
        border_fraction = 0.07
        replacement_val = 0
        y_border_px = int(border_fraction * cell_img.shape[0])
        x_border_px = int(border_fraction * cell_img.shape[1])
        cell_img[:, 0:x_border_px] = replacement_val
        cell_img[:, -x_border_px:] = replacement_val
        cell_img[0:y_border_px, :] = replacement_val
        cell_img[-y_border_px:, :] = replacement_val
    contours = cv2.findContours(image=cell_img,
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)        
        largest_contour_area = cv2.contourArea(contours[0])
        image_area = cell_img.shape[0] * cell_img.shape[1]
        contour_percentage_area = 100 * largest_contour_area / image_area
        if contour_percentage_area > area_threshold:
            image_contains_digit = True
        else:
            image_contains_digit = False
    else:
        image_contains_digit = False
    return image_contains_digit, cell_img


def locate_cells(grid_img):
    valid_cells = []
    grid_area = grid_img.shape[0] * grid_img.shape[1]
    grid_img = to_grayscale(grid_img, method="mean", blocksize=91, c=7)
    contours = cv2.findContours(image=grid_img.copy(),
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_NONE)
    if contours:
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            contour_fractional_area = cv2.contourArea(contour) / grid_area
            if len(approx) == 4 and contour_fractional_area > 0.005 and contour_fractional_area < 0.015:
                mask = np.zeros_like(grid_img)
                cv2.drawContours(image=mask,
                                contours=[contour],
                                contourIdx=0,
                                color=255,
                                thickness=cv2.FILLED)
                y_px, x_px = np.where(mask==255)
                cell_image = grid_img[min(y_px):max(y_px)+1, min(x_px):max(x_px)+1]
                digit_is_present, cell_image = detect_digit(img=cell_image,
                                                                             area_threshold=5,
                                                                             apply_border=True)
                kernel = np.ones((3, 3), np.uint8)
                cell_image = cv2.erode(cell_image, kernel, iterations=1)
                cell_image = cv2.resize(cell_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
                moments = cv2.moments(contour)
                x_centroid = int(moments['m10'] / moments['m00'])
                y_centroid = int(moments['m01'] / moments['m00'])
                valid_cells.append({'img': cell_image,
                                    'contains_digit': digit_is_present,
                                    'x_centroid': x_centroid,
                                    'y_centroid': y_centroid})
    else:
        print("No valid cells found in image")
    return valid_cells


def extract_valid_cells(img):
    M_matrices, warped_images = find_grid_candidates(img)
    if not warped_images:
        raise Exception("No grid candidates were found in the image.")
    for i, grid_image in enumerate(warped_images):
        valid_cells = locate_cells(grid_image)
        M = M_matrices[i]
        if len(valid_cells) == 81:
            valid_cells = sort_cells(valid_cells)
            return valid_cells, M, grid_image
    raise Exception("Unable to find the required number of cells in image.")


def sort_cells(cells):
    x_vals = [cell['x_centroid'] for cell in cells]
    y_vals = [cell['y_centroid'] for cell in cells]
    points = np.array([[cell['x_centroid'], cell['y_centroid']] for cell in cells])
    points_sorted = np.array(sorted(points, key=lambda x: x[1]))
    rows = np.reshape(points_sorted, newshape=(9, 9, 2))
    final = np.array([sorted(row, key=lambda x: x[0]) for row in rows])
    final_reshaped = np.reshape(final, newshape=(81, 2))
    for i in range(len(x_vals)):
        assert any(np.equal(final_reshaped, [x_vals[i], y_vals[i]]).all(1))
    indices = []
    for x, y in final_reshaped:
        x_indices = np.where(np.array(x_vals) == x)
        y_indices = np.where(np.array(y_vals) == y)
        index = np.intersect1d(x_indices, y_indices)[0]
        indices.append(index)
    sorted_cells_list = [cells[idx] for idx in indices]
    return sorted_cells_list


def predict_grid(model, cells):
    digit_images = np.array([np.expand_dims(cell['img'], -1) for cell in cells if cell['contains_digit']])
    pred_labels = model.predict(digit_images)
    pred_labels = np.argmax(pred_labels, axis=1) + 1
    indices = np.where([cell['contains_digit'] for cell in cells])[0]
    grid_array = np.zeros((81), dtype=int)
    grid_array[indices] = pred_labels
    grid_array = np.reshape(grid_array, newshape=(9, 9))
    return grid_array


def render_solution(full_image, board_image, cells_list, solved_board_arr, M_matrix):
    font = cv2.FONT_HERSHEY_SIMPLEX
    solution_img = np.ones_like(board_image) * 255
    flattened_board_array = solved_board_arr.reshape((-1))
    for i, cell in enumerate(cells_list):
        if not cell['contains_digit']:
            x_pos = cell['x_centroid']
            y_pos = cell['y_centroid']
            text = str(flattened_board_array[i])
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = int((x_pos - textsize[0] / 2))
            text_y = int((y_pos + textsize[1] / 2))
            cv2.putText(solution_img, text, (text_x, text_y), font, 1.3, (0, 0, 0), 2)
    unwarped_img = cv2.warpPerspective(
        solution_img,
        M_matrix,
        (full_image.shape[1], full_image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    annotated = full_image.copy()
    annotated[np.where(unwarped_img[:,:,0] == 0)] = (255, 15, 0)
    return annotated