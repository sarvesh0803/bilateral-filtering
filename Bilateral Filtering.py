import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma**2))) * np.exp(-(x**2) / (2 * sigma**2))


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# sigma_r is the standard deviation for range filtering
# sigma_d is the standard deviation for domain filtering
def apply_filter(image, x, y, sigma_r, sigma_d):
    sum_of_weights = 0.0
    filtered_intensity = 0.0

    # perform filtering using a mask of 5x5
    for i in range(5):
        for j in range(5):
            n_x = x + 2 - i
            n_y = y + 2 - j

            # calcuate photometric weights and geometric weights
            weight_range = gaussian(image[n_x][n_y] - image[x][y], sigma_r)
            weight_domain = gaussian(
                euclidean_distance(x, y, n_x, n_y), sigma_d)

            weight = weight_range * weight_domain

            filtered_intensity += image[n_x][n_y] * weight
            sum_of_weights += weight

    # update the output image
    filtered_intensity = filtered_intensity / sum_of_weights

    return filtered_intensity


def bilateral_filter(source, sigma_r, sigma_d):
    # pad the image with a border of 2 pixels
    s = cv2.copyMakeBorder(
        source, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255))

    # initialize the output container
    output = s - s
    output = output.astype('float64')
    output.setflags(write=True)

    # apply filter over each pixel
    for i in range(2, len(s) - 1):
        for j in range(2, len(s[0]) - 1):
            output[i][j] = apply_filter(s, i, j, sigma_r, sigma_d)

    # delete the padded borders
    output = np.delete(
        output, [0, 1, len(output[0]) - 1,
                 len(output[0]) - 2], axis=1)
    output = np.delete(
        output, [0, 1, len(output) - 1, len(output) - 2], axis=0)

    return output


# image affected by salt and pepper noise only
spnoisy = cv2.imread("/home/sarvesh/Projects/Image Processing/spnoisy.jpg", 0)
spnoisy = spnoisy.astype('float64')
# image affected by uniform noise only
unifnoisy = cv2.imread("/home/sarvesh/Projects/Image Processing/unifnoisy.jpg",
                       0)
unifnoisy = unifnoisy.astype('float64')
# image affected by uniform and salt & pepper noise both
spunifnoisy = cv2.imread(
    "/home/sarvesh/Projects/Image Processing/spunifnoisy.jpg", 0)
spunifnoisy = spunifnoisy.astype('float64')


# define the standard deviations for range
r_sigmas = [10, 30, 100, 300]

# define the standard deviations for domain
d_sigmas = [1, 3, 10]

for r in r_sigmas:
    for d in d_sigmas:
        # perform bilateral filtering for each type of image and save the image
        spnoisy_filtered = bilateral_filter(spnoisy, r, d)
        name = "spnoisy_filtered_" + str(r) + "_" + str(d)
        cv2.imwrite(name + ".jpeg", spnoisy_filtered)

        unifnoisy_filtered = bilateral_filter(unifnoisy, r, d)
        name = "unifnoisy_filtered_" + str(r) + "_" + str(d)
        cv2.imwrite(name + ".jpeg", unifnoisy_filtered)

        spunifnoisy_filtered = bilateral_filter(spunifnoisy, r, d)
        name = "spunifnoisy_filtered_" + str(r) + "_" + str(d)
        cv2.imwrite(name + ".jpeg", spunifnoisy_filtered)