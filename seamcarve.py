import cv2
import numpy as np

import sys

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0, 0)

def x_gradient(img):
    gradient = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3,
                         scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return gradient

def y_gradient(img):
    gradient =  cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                          scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return gradient

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def energy(img):
    blurred = gaussian_blur(img)
    gray = grayscale(blurred)
    dx = x_gradient(gray)
    dy = y_gradient(gray)

    return cv2.add(np.absolute(dx), np.absolute(dy))

def cumulative_energies(energy):
    height, width = energy.shape[:2]
    energies = np.zeros((height, width))

    for i in xrange(1, height):
        for j in xrange(width):
            previous = []
            previous.append(energies[i - 1, j])

            if j - 1 >= 0:
                previous.append(energies[i - 1, j - 1])

            if j + 1 < width:
                previous.append(energies[i - 1, j + 1])

            energies[i, j] = energy[i, j] + min(previous)

    return energies


def get_min_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []

    for i in xrange(height - 1, -1, -1):
        row = energies[i, :]

        if i == height - 1:
            previous = np.argmin(row)
            seam.append([previous, i])
        else:
            left = row[previous - 1] if previous - 1 >= 0 else 1e6
            middle = row[previous]
            right = row[previous + 1] if previous + 1 < width else 1e6

            previous = previous + np.argmin([left, middle, right]) - 1
            seam.append([previous, i])

    return seam

def draw_seam(img, seam):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    cv2.imshow('seam', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_seam(img, seam):
    height, width, bands = img.shape
    removed = np.zeros((height, width - 1, bands), np.uint8)

    for x, y in reversed(seam):
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]

    return removed

def resize(img):
    result = img

    for i in xrange(50):
        energies = cumulative_energies(energy(result))
        seam = get_min_seam(energies)

        draw_seam(result, seam)

        result = remove_seam(result, seam)

    cv2.imshow('removed', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    # energies = cumulative_energies(energy(img))
    # seam = get_min_seam(energies)
    # draw_seam(img, seam)
    #
    # removed = remove_seam(img, seam)
    resize(img)

    # cv2.imshow('removed', removed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
