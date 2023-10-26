# Built in packages
import math
import sys
from pathlib import Path

#Extensions for the added functionality
import numpy as np
from pyzbar import pyzbar

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:

def convertRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    row = 0
    counter = 0
    for i in range((image_width) * (image_height-1)):
        greyscale_pixel_array[row][i%image_width] = round(0.299 * px_array_r[row][i%image_width] + 0.587 * px_array_g[row][i%image_width] + 0.114 * px_array_b[row][i%image_width])
        if i % image_width == 0 and row != 0: #need to ensure that the 756 0's row is counted 
            row += 1
        if row == 0:
            counter += 1
        if counter == 755 and row == 0:
            row += 1
    return greyscale_pixel_array

def countZeroes(arr):
    arr = [x for x in arr if x == 0]
    print(len(arr))


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = pixel_array[0][0]
    max_value = pixel_array[0][0]
    for i in range(len(pixel_array)):
        for x in range(len(pixel_array[i])):
            if pixel_array[i][x] > max_value:
                max_value = pixel_array[i][x]
            elif pixel_array[i][x] < min_value:
                min_value = pixel_array[i][x]
    return [min_value, max_value]

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    greyscale_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min_max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    min_val = min_max[0]
    max_val = min_max[1]
    g_min = 0
    g_max = 255
    if (max_val - min_val) == 0:
        fractional = 0
    else:
        fractional = (g_max-g_min) / (max_val - min_val)
    for i in range(len(pixel_array)):
        for x in range(len(pixel_array[i])):
            pixel = pixel_array[i][x]
            s_out = round((pixel - min_val)*(fractional) + g_min)
            if s_out < g_min:
                greyscale_array[i][x] = g_min
            elif g_min <= s_out <= g_max:
                greyscale_array[i][x] = s_out
            elif s_out > g_max:
                greyscale_array[i][x] = g_max
    return greyscale_array
    
def computeVerticalEdgesSobel(pixel_array, image_width, image_height):
    
    edge_array = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_kernel = [[-1, 0, 1],[-2,0,2],[-1, 0, 1]]
    
    for y in range(0, image_height):
        for x in range(0, image_width):
            if (y==0) or (y==len(pixel_array)-1):
                edge_array[y][x] = 0.0
            elif (x==0) or (x==len(pixel_array[y])-1):
                edge_array[y][x] = 0.0
            else:
                neighbourhood_1 = pixel_array[y-1][x-1:x+2]
                neighbourhood_2 = pixel_array[y][x-1:x+2]
                neighbourhood_3 = pixel_array[y+1][x-1:x+2]
                gradient_1 = sum([neighbourhood_1[i]*sobel_kernel[0][i] for i in range(3)])
                gradient_2 = sum([neighbourhood_2[i]*sobel_kernel[1][i] for i in range(3)])
                gradient_3 = sum([neighbourhood_3[i]*sobel_kernel[2][i] for i in range(3)])
                
                edge_array[y][x] = (gradient_1 + +gradient_2 + gradient_3) * 0.125
            
    return edge_array

def computeHorizontalEdgesSobel(pixel_array, image_width, image_height):
    
    edge_array = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_kernel = [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]
    
    for y in range(0, image_height):
        for x in range(0, image_width):
            if (y==0) or (y==len(pixel_array)-1):
                edge_array[y][x] = 0.0
            elif (x==0) or (x==len(pixel_array[y])-1):
                edge_array[y][x] = 0.0
            else:
                neighbourhood_1 = pixel_array[y-1][x-1:x+2]
                neighbourhood_2 = pixel_array[y][x-1:x+2]
                neighbourhood_3 = pixel_array[y+1][x-1:x+2]
                gradient_1 = sum([neighbourhood_1[i]*sobel_kernel[0][i] for i in range(3)])
                gradient_2 = sum([neighbourhood_2[i]*sobel_kernel[1][i] for i in range(3)])
                gradient_3 = sum([neighbourhood_3[i]*sobel_kernel[2][i] for i in range(3)])
                
                edge_array[y][x] = abs(gradient_1 + +gradient_2 + gradient_3) * 0.125
            
    return edge_array

def computeSobelDiff(xDirectionArray, yDirectionArray, image_width, image_height):

    edge_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(0, image_height):
        for x in range(0, image_width):
            edge_array[y][x] = abs(xDirectionArray[y][x]-yDirectionArray[y][x])
    
    return edge_array

def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    
    edge_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    top_border = [x for x in pixel_array[0]]
    top_border.insert(0, pixel_array[0][0])
    top_border.append(pixel_array[0][-1])
    bottom_border = [n for n in pixel_array[-1]]
    bottom_border.insert(0, pixel_array[-1][0])
    bottom_border.append(pixel_array[-1][-1])
    for i in range(len(pixel_array)):
        pixel_array[i].insert(0, pixel_array[i][0])
        pixel_array[i].append(pixel_array[i][-1])
    pixel_array.insert(0, top_border)
    pixel_array.append(bottom_border)
    
    gaussian = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    
    for y in range(1, image_height+1):
        for x  in range(1, image_width+1):
            neighbourhood_1 = pixel_array[y-1][x-1:x+2]
            neighbourhood_2 = pixel_array[y][x-1:x+2]
            neighbourhood_3 = pixel_array[y+1][x-1:x+2]
            
            gradient_1 = sum([neighbourhood_1[i]*gaussian[0][i] for i in range(3)])
            gradient_2 = sum([neighbourhood_2[i]*gaussian[1][i] for i in range(3)])
            gradient_3 = sum([neighbourhood_3[i]*gaussian[2][i] for i in range(3)])
            
            gradient = abs(gradient_1+gradient_2+gradient_3) / 16
            edge_array[y-1][x-1] = gradient
    return edge_array
            
def computeThreshold(pixel_array, threshold_value, image_width, image_height):
    for y in range(0, image_height):
        for x in range(0, image_width):
            if pixel_array[y][x] < threshold_value:
                pixel_array[y][x] = 0.0
            else:
                pixel_array[y][x] = 255.0
    return pixel_array

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    greyscaleArray = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(len(pixel_array)):
        pixel_array[i].insert(0, 0)
        pixel_array[i].append(0)
    
    pixel_array.insert(0, [0 for x in range(image_width+2)])
    pixel_array.append([0 for x in range(image_width+2)])

    for y in range(1, len(pixel_array)-1):
        for x in range(1, len(pixel_array[y])-1):
            top_row = [pixel_array[y-1][x-1], pixel_array[y-1][x], pixel_array[y-1][x+1]]
            middle_row = [pixel_array[y][x-1], pixel_array[y][x], pixel_array[y][x+1]]
            bottom_row = [pixel_array[y+1][x-1], pixel_array[y+1][x], pixel_array[y+1][x+1]]
            if 0 not in (top_row+middle_row+bottom_row):
                greyscaleArray[y-1][x-1] = 1
    return greyscaleArray

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    greyscaleArray = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(len(pixel_array)):
        pixel_array[i].insert(0, 0)
        pixel_array[i].append(0)
    
    pixel_array.insert(0, [0 for x in range(image_width+2)])
    pixel_array.append([0 for x in range(image_width+2)])

    for y in range(1, len(pixel_array)-1):
        for x in range(1, len(pixel_array[y])-1):
            top_row = [pixel_array[y-1][x-1], pixel_array[y-1][x], pixel_array[y-1][x+1]]
            middle_row = [pixel_array[y][x-1], pixel_array[y][x], pixel_array[y][x+1]]
            bottom_row = [pixel_array[y+1][x-1], pixel_array[y+1][x], pixel_array[y+1][x+1]]
            if sum(top_row+middle_row+bottom_row) != 0:
                greyscaleArray[y-1][x-1] = 1
    return greyscaleArray

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):

    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    component = createInitializedGreyscalePixelArray(image_width, image_height)
    label = 1
    a_dict = {}
    
    for y in range(len(pixel_array)):
        for x in range(len(pixel_array[y])):
            if pixel_array[y][x] != 0 and visited[y][x] == 0:
                pixelQueue = Queue()
                pixelQueue.enqueue((x, y))
                visited[y][x] = 1
                while not pixelQueue.isEmpty():
                    currentPix = pixelQueue.dequeue()
                    currentX = currentPix[0]
                    currentY = currentPix[1]
                    visited[currentY][currentX] = 1
                    component[currentY][currentX] = label
                    try:
                        a_dict[label] += 1
                    except:
                        a_dict[label] = 1
                    if (currentX-1 >= 0):
                        if (pixel_array[currentY][currentX-1] != 0 and visited[currentY][currentX-1] == 0):
                            pixelQueue.enqueue((currentX-1, currentY))
                            visited[currentY][currentX-1] = 1
                    if (currentX+1 <= image_width-1):
                        if (pixel_array[currentY][currentX+1] != 0 and visited[currentY][currentX+1] == 0):
                            pixelQueue.enqueue((currentX+1, currentY))
                            visited[currentY][currentX+1] = 1
                    if (currentY-1 >= 0):
                        if (pixel_array[currentY-1][currentX] != 0 and visited[currentY-1][currentX] == 0):
                            pixelQueue.enqueue((currentX, currentY-1))
                            visited[currentY-1][currentX] = 1
                    if (currentY+1 <= image_height-1):
                        if (pixel_array[currentY+1][currentX] != 0 and visited[currentY+1][currentX] == 0):
                            pixelQueue.enqueue((currentX, currentY+1))
                            visited[currentY+1][currentX] = 1
                label += 1
            else:
                visited[y][x] = 1

    return (component, a_dict)

def computeGreyscaleBarcode(pixel_array, image_width, image_height, min_x, max_x, min_y, max_y):
    height_array = pixel_array[min_y:max_y]
    width_array = [col[min_x:max_x] for col in height_array]
    return width_array


# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode2"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    greyscaleArrayPre = convertRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    greyscaleArrayPost = scaleTo0And255AndQuantize(greyscaleArrayPre, image_width, image_height)

    xDirectionArray = computeVerticalEdgesSobel(greyscaleArrayPost, image_width, image_height)
    yDirectionArray = computeHorizontalEdgesSobel(greyscaleArrayPost, image_width, image_height)
    filteredArray = computeSobelDiff(xDirectionArray, yDirectionArray, image_width, image_height)
    for i in range(20):
        filteredArray = computeGaussianAveraging3x3RepeatBorder(filteredArray, image_width, image_height)

    computeThreshold(filteredArray, 28.5, image_width, image_height)

    for z in range(4):
        filteredArray = computeDilation8Nbh3x3FlatSE(filteredArray, image_width, image_height)
    for n in range(2):
        filteredArray = computeErosion8Nbh3x3FlatSE(filteredArray, image_width, image_height)

    componentTuple = computeConnectedComponentLabeling(filteredArray, image_width, image_height)

    finalImage = componentTuple[0]
    finalDict = componentTuple[1]


    min_x = image_width - 1
    max_x = 0
    min_y = image_height - 1
    max_y = 0
    
    print(finalDict)
    for label in finalDict:
        for y in range(len(finalImage)):
            for x in range(len(finalImage[y])):
                if finalImage[y][x] == label:
                    if x < min_x:
                        min_x=x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        box_width = max_x - min_x
        box_height = max_y - min_y
        if (max([box_height, box_width]) <= (1.8 * min([box_height, box_width]))) and (finalDict[label] > (image_width*image_height)/80):
            break
        min_x = image_width - 1
        max_x = 0
        min_y = image_height - 1
        max_y = 0

    adjusted_min_x = round(min_x - (min_x / 80))
    adjusted_max_x = round(max_x + (max_x / 80))
    adjusted_min_y = round(min_y - (min_y / 80))
    adjusted_max_y = round(max_y + (max_y / 80))

    barcode_scan = computeGreyscaleBarcode(greyscaleArrayPost, image_width, image_height, min_x, max_x, min_y, max_y)
    rgb_width = len(barcode_scan[0])
    rgb_height = len(barcode_scan)
    rgb_image = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = barcode_scan
    rgb_image[:, :, 1] = barcode_scan
    rgb_image[:, :, 2] = barcode_scan

    barcodes = pyzbar.decode(rgb_image)

    try:
        print("The code for this barcode is", barcodes[0].data, "and the barcode type is", barcodes[0].type)
    except:
        print("Data cannot be extracted for this barcode.")

    px_array = px_array_r


    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    center_x = ((adjusted_max_x+adjusted_min_x)/2)
    center_y = ((adjusted_min_y+adjusted_max_y)/2)
    bbox_min_x = adjusted_min_x
    bbox_max_x = adjusted_max_x
    bbox_min_y = adjusted_min_y
    bbox_max_y = adjusted_max_y

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()