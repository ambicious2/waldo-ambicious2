import sys
from PIL import Image
import cv2

""" 
  I took the approach to this challenge as a prototype; therefore I have made assumptions and since the scope is
 small and limited, I have separated concerns into methods/functions rather than individual classes.
 
 Example CL: python3 images.py ./appleBananaPeach.jpg ./apple.jpeg
 
 Assumptions: 
    - Open source usage without concern for licensing
    
 Technologies and Libraries used:
    - Python (3.61)
    - Pillow (5.2.0)
    - OpenCV (opencv-python 3.4.2.17)
    - Numpy (1.15.0)
"""

def naiveComparison(img1, img2):
    """
    Naive comparison of images to determine which image is bigger (quick rough-cut check to determine and return
    the image that would possibly contain the other).
    :param img1:
    :param img2:
    :return:
    """
    img1Size = 1
    if img1.size[0] > 0:
        img1Size = img1.size[0]
    if img1.size[1] > 0:
        img1Size *= img1.size[1]

    img2Size = 1
    if img2.size[0] > 0:
        img2Size = img2.size[0]
    if img2.size[1] > 0:
        img2Size *= img2.size[1]

    return img1 if img1Size > img2Size else img2


def calculateHash(image):
    """
    Compute the difference in brightness between adjacent pixels and identify the relative gradient direction.

    :param image: Multimedia file to capture a fingerprint from

    :return: String representing the hexadecimal value

    """

    # NTS: convert this to a parameter to allow for flexibility
    hashSize = 8

    # Convert the image to Grayscale and shrink it to a common size
    #
    # If we grayscale the image, we reduce each pixel value to a luminous intensity value; and shrinking the
    # image to a common base size allows us to remove all high frequencies and details (noise) giving us a sample
    # size of 72 intensity values.
    #
    image = image.convert('L').resize(
        (hashSize + 1, hashSize),
        Image.ANTIALIAS,
    )

    # Compare adjacent pixels.
    #
    # Compare each intensity value in the list to it's adjacent value for each row, resulting in array of 1's or 0's
    #
    pixels = list(image.getdata())
    difference = []
    for row in range(hashSize):
        for col in range(hashSize):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            difference.append(pixel_left > pixel_right)

            #
            # Convert the binary array to a hexadecimal string.
            #
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                    decimal_value = 0
    return ''.join(hex_string)


def calculateHammingDistance(s1, s2):
    """
      For speed and quickly get up and running, I used the code found on Wikipedia.
      Credit: https://en.wikipedia.org/wiki/Hamming_distance#Algorithm_example
    """
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def quickImageAnalysis(imageTuple):
    """ Quick hash calculation/comparison. Hashes can be stored in a datastore in the future """
    hashForImage1 = calculateHash(imageTuple[0][0])
    hashForImage2 = calculateHash(imageTuple[1][0])

    hammingDistance = calculateHammingDistance(hashForImage1, hashForImage2)

    # Can fine tune here for leniency
    if hammingDistance == 0:
        print('Images are the same.')
        exit(0)

    biggerImage = naiveComparison(imageTuple[0][0], imageTuple[1][0])
    if biggerImage == imageTuple[1][0]:
        return (imageTuple[0][1], imageTuple[1][1])
    else:
        return (imageTuple[1][1], imageTuple[0][1])


def inputValidation():
    if (len(sys.argv) != 3):
        print('Sorry, you need to pass in the location of two image files for comparison')
        exit(1)
    image1 = Image.open(sys.argv[1])
    if image1.mode != "RGB":
        print('Sorry, the first image is not a JPEG')
        exit(1)
    image2 = Image.open(sys.argv[2])
    if image2.mode != "RGB":
        print('Sorry, the second image is not a JPEG')
        exit(1)
    return ((image1, sys.argv[1]), (image2, sys.argv[2]))

def imageComposition(sourceTuple):
    """
    Use computer vision for object detection. The algorithm can be fine-tuned for anomalies, however due to time
    constraints, this is a rough draft (first cut iteration).

    Credit(s):
      https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#cv2.matchTemplate
      https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_table_of_contents_objdetect/py_table_of_contents_objdetect.html#py-table-of-content-objdetection
      https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

    :param sourceTuple:
    :return:
    """
    smallImage = cv2.imread(sourceTuple[0])
    largeImage = cv2.imread(sourceTuple[1])
    result = cv2.matchTemplate(smallImage, largeImage, cv2.TM_SQDIFF_NORMED)

    # We want the minimum squared difference
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(result)

    # Extract the coordinates of our best match
    # Since we are using TM_SQDIFF_NORMED, we use the minimum
    minLocationX, minLocationY = minLocation

    # Get the size of the template. This is the same size as the match.
    width, height = smallImage.shape[:2]

    """
    # Draw the rectangle

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(largeImage, (minLocationX, minLocationY), (minLocationX + height, minLocationY + width), (0, 0, 255), 2)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output',largeImage)

    # The image is only displayed if we call this
    cv2.waitKey(0)
    """

    location = ''
    if minLocationY < (largeImage.shape[1] / 2):
        location += 'upper'
    else:
        location += 'lower'
    location += ' '
    if (minLocationX + width) < (largeImage.shape[0] / 2):
        location += 'left'
    else:
        location += 'right'
    return location



# Start of script
#
imageTuple = inputValidation()
sourceTuple = quickImageAnalysis(imageTuple)
print('Found location at: ', imageComposition(sourceTuple))


