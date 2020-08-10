import cv2
import numpy as np
import os

ans = input("Enter the name of your document ")

if not os.path.exists(ans):
    print("File not found , please enter proper name !")
    exit()

kernel = np.ones((5, 5))
img = cv2.imread(ans)

cv2.Laplacian(img, cv2.CV_64F).var()
img = cv2.resize(img, (1000, 1000))
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def check_blurry(temp_img):
    fm = cv2.Laplacian(temp_img, cv2.CV_64F).var()
    text = "Not Blurry"
    if fm < 200:
        text = "Blurry"
    return text


if check_blurry(grey_img) == "Blurry":
    threshold = 50
else:
    threshold = 200

blur_img = cv2.GaussianBlur(grey_img, (5, 5), 1)
canny = cv2.Canny(blur_img, threshold, threshold)
dilated = cv2.dilate(canny, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)


def set_order(approx_points):
    approx_points = approx_points.reshape(4, 2)
    approx_new_pts = np.zeros((4, 2))
    add_x_y = approx_points.sum(axis=1)

    approx_new_pts[0] = approx_points[np.argmin(add_x_y)]
    approx_new_pts[3] = approx_points[np.argmax(add_x_y)]

    diff_x_y = np.diff(approx_points, axis=1)

    approx_new_pts[1] = approx_points[np.argmin(diff_x_y)]
    approx_new_pts[2] = approx_points[np.argmax(diff_x_y)]
    return approx_new_pts


def get_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, )
    temp_area = 0
    flag = found_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if area > temp_area:
            temp_area = area
            temp_approx = approx
            found_area = 1

    if found_area == 1:
        if len(temp_approx) == 4:
            flag = 1

    if flag == 0:
        print("Could not find the prominent document , please upload a clearer picture.")
        exit()

    approx = temp_approx
    approx = set_order(approx)

    return approx


def increase_brightness(img):
    alpha = 1
    beta = 20
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def warp_document(approx):
    points1 = np.float32(approx)  # cropped part from original image
    points2 = np.float32([[0, 0], [500, 0], [0, 700], [500, 700]])  # size to change cropped part into
    matrix = cv2.getPerspectiveTransform(points1, points2)  # assertion of cropped into desired size
    final_doc = cv2.warpPerspective(img, matrix, (500, 700))
    cv2.imshow('DOCUMENT', final_doc)
    cv2.waitKey(0)

    check = input("Do you want to increase the brightness and contrast of image? (y/n)")
    if check == 'y':
        final_doc = increase_brightness(final_doc)
        cv2.imshow("with increased brightness and contrast", final_doc)

    cv2.imwrite("new_" + ans, final_doc)
    print("Your new file is saved as 'new_" + ans, "'")


approximated_points = get_contours(eroded)
warp_document(approximated_points)
cv2.waitKey(0)
