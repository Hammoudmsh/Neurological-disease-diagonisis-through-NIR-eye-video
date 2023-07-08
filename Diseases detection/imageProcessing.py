import cv2
import numpy as  np
import time
from PIL import Image
from PIL import  ImageGrab

from PIL import Image, ImageChops
from matplotlib import animation as anim
import matplotlib.pyplot as plt
import matplotlib




net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")


def saveImg(name,image):
    cv2.imwrite(name + ".png", image)


def maxPooling(img,k):
    dst = img.copy()

    w,h,c = img.shape
    #Length from the center pixel to both ends pixels
    size = k // 2

    #Pooling process
    for x in range(size, w, k):
        for y in range(size, h, k):
            dst[x-size:x+size,y-size:y+size,0] = np.min(img[x-size:x+size,y-size:y+size,0])
            dst[x-size:x+size,y-size:y+size,1] = np.min(img[x-size:x+size,y-size:y+size,1])
            dst[x-size:x+size,y-size:y+size,2] = np.min(img[x-size:x+size,y-size:y+size,2])

    return dst

def fill_holes1(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def fill_holes(img, expr=lambda x: x > 0):
    img_binary = 255 * expr(img).astype(np.uint8)
    img_tmp = img_binary.copy()
    h, w, _ = img_tmp.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(img_tmp, mask, (0, 0), 255)
#     cv2.floodFill(img_tmp, mask, (0, 0), 255)
    img_fill = np.bitwise_or(img_binary, ~img_tmp)

    return img_fill

def info(videoName):
    cap = cv2.VideoCapture(videoName)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    minutes = int(duration/60)
    seconds = duration%60
    delayTime = duration/frame_count
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    cap.release()
    return fps, frame_count, minutes, seconds,duration, delayTime, width, height


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = 200#min(y, x)
    start_x = (x // 2) - (min_dim // 2)

    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def plot(title, rows, columns, figsize, titles,*Images):
    # create figure
    fig = plt.figure(figsize = figsize,)
    plt.title(title)
    plt.axis('off')

#     matplotlib.rcParams['axes.titlepad'] = 0
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.pyplot.subplots_adjust(wspace=1, hspace=1)

    for i in range(1, len(Images)+1):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i)
        # showing image
        if Images[i-1] is not None:
            plt.imshow(Images[i-1])
            plt.title(titles[i-1])
        else:
            pass
        plt.axis('off')
        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.3,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.1)

    return fig

def showImages(lables,nr, nc, dimension, *imList):

#     if not debug:
#         return

    imageList = imList

    images = []
    row = []

    lablesNew = []
    rowLabel = []

    for r in range(0, len(imageList)):
        img = cv2.resize(imageList[r],dimension,fx=0.5, fy=0.5)
        #cv2.putText(img,lables[i], (20, 20),cv2.FONT_HERSHEY_PLAIN ,2,(255,255,0),2) # ADD THE GRADE TO NEW IMAGE
        row.append(img)
        rowLabel.append(lables[r])


        if (r+1) % nc == 0:
            images.append(row)
            lablesNew.append(rowLabel)
            row = []
            rowLabel = []
    stackedImage = stackImages(images,0.5,lablesNew)
    return stackedImage

#-----------------------------------------------------------------------------------------------------
    ## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                if lables[d][c]!="":
                    #cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                    cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,255),1)
    return ver


def show(image: np.array) -> None:
    """
    Plot image
    :param image: Image to plot
    :param size:
    :return: Nothing
    """
    return Image.fromarray(image)

def getGazeCoordinates(videoName, show = True, write_movie = True, method = "hough"):
    alpha=4.9
    beta=10
    coor2save = []

    try:
        fps, frame_count, minutes, seconds,duration, delayTime,  width, height = info(videoName)
        cap = cv2.VideoCapture(videoName)
    except:
        print(videoName)
        return []

    while (cap.isOpened()):
        success, frame = cap.read()
        if success:
            frame=cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
#             frame = increaseContrast(frame.copy(), [50,50,50])
            frame = gamma_correction(byte2float(frame),6)
            result, tmp = circleDetection(frame, method)
            coor2save.append(tmp[0])
            if show:
                cv2.imshow(method + " method", result)

                if (cv2.waitKey(1) & 0xFF) == ord('q') or  (cv2.waitKey(33) == 27):
                    cap.release()
                    break
                time.sleep(delayTime)
        else:
            cap.release()
            break
    if show:
        cv2.destroyAllWindows()

    cap.release()
    return coor2save



def finCircleContour(canny_img):
    img = canny_img.copy()
    data = []
    contours, thr = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        appox = cv2.approxPolyDP(cnt, .03*cv2.arcLength(cnt, True), True)
        if len(appox)==8:
            area = cv2.contourArea(cnt)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circleArea = radius * radius * np.pi
            #if circleArea == area:
            if cv2.isContourConvex(appox):
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
                data.append([cx, cy, radius])

    return img, data




def HED(image):
    # global net
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    return hed


lastpos = {"left":[0,0,0], "right":[0,0,0]}

def houghTransform(img):
    coor =[]
    left = []
    right = []

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(img,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 10,
                param2 = 30, minRadius = 1, maxRadius = 100)

    # Draw circles that are detected.

    x = img#gray_blurred.copy()

    if detected_circles is None:
        x, lastpos['left'], lastpos['right'] = drawGazes(x.copy(),lastLeft = lastpos['left'], lastRight = lastpos['right'])
    else:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        if len(detected_circles[0, :]) >= 1:
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                if a < 150:
                    left = [a, b, r]
                else:
                    right = [a, b, r]
            x, lastpos['left'], lastpos['right'] = drawGazes(x.copy(),
                                                               left,
                                                               right,
                                                               lastLeft = lastpos['left'],
                                                               lastRight =lastpos['right'])

    coor.extend(lastpos['left'])
    coor.extend(lastpos['right'])
    #mask = np.ones(shape=gray_blurred.shape[0:2], dtype = "bool")
    #ImageChops.logical_xor(mask, gray_blurred)
    #u = 255 - np.bitwise_xor(mask, gray_blurred).astype(np.uint8)
    return  x , coor


def drawGazes(img, left =None, right = None, lastLeft = None, lastRight = None, color = (0, 0, 0)):
    if left:
        al, bl, rl = left
        lastLeftNew = left
    else:
        al, bl, rl = lastLeft
        lastLeftNew = lastLeft

    if right:
        ar, br, rr = right
        lastRightNew =  right
    else:
        ar, br, rr = lastRight
        lastRightNew =  lastRight


    mask = img#255 * np.ones(shape = img.shape[0:2], dtype = "uint8")
    #cv2.circle(img = mask, center = (al, bl), radius = 15, color = color, thickness=-1)
    #cv2.circle(img = mask, center = (ar, br), radius = 15, color = color, thickness=-1)
#     cv2.circle(mask, (al, bl), 3, (0, 255, 0), -1)
#     cv2.circle(mask, (ar, br), 3, (0, 255, 0), -1)
    return mask, lastLeftNew, lastRightNew




def circleDetection(image,method):
    coor =[]

    img = image.copy()
    grayImg = rgb2gray(img)

    if method == "canny":
        res = cannyFilter(image.copy(), (25,25), [95,120], (30,30), 20, 20)
        coor = [1,1,1,1,1,1]
    elif method == "hsv":
        res = maskHsv(image.copy(), down = (0, 0, 0), up = (255, 255, 255))
        res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
#         res, coor = finCircleContour(res)

        #         res = maxPooling(res, 2)
        coor = [1,1,1,1,1,1]
    elif method == "contours":
        res, data2 = finCircleContour(grayImg)
        coor = [1,1,1,1,1,1]
    elif method == "hough":
        #gray_blurred = cv2.blur(grayImg, (3,3))
        gray_blurred = bilateralFilter(img = grayImg)
#         gray_blurred = closeImg(gray_blurred, (300,300))
#         gray_blurred[gray_blurred>60] = 255
#         gray_blurred[gray_blurred<60] = 0
#         img = gamma_correction(byte2float(img),10)
        res, coor = houghTransform(gray_blurred)
        #res = cannyFilter(res.copy(), (5,5), [20,100], (5,5), 5, 5)
    elif method == "hed":
        res = HED(img.copy())
        coor =[1,1,1,1,1,1]

    al, bl,rl, ar, br,rr = coor
    cv2.circle(img, (al, bl), 3, (0, 255, 0), -1)
    cv2.circle(img, (ar, br), 3, (0, 255, 0), -1)
    return res,coor


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def gray2bin(img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


def dilate(binImg, kernelParam, dp_num):
    kernel = np.ones(kernelParam)
    imgDial = cv2.dilate(binImg, kernel, iterations= dp_num) # APPLY DILATION

def erode(binImg, kernelParam, ep_num):
    kernel = np.ones(kernelParam)
    imgDial = cv2.erode(binImg, kernel, iterations= ep_num) # APPLY DILATION

def openImg(img, kernel):
    return  cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closeImg(img, kernel):
    return  cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def TopHatFilter(img, filterSize =(3, 3)):
    img = self.rgb2gray(img)
    # Getting the kernel to be used in Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)

    # Applying the Top-Hat operation
    topHatImg = cv2.morphologyEx(img,
                                  cv2.MORPH_TOPHAT,
                                  kernel)
    return tophatImg

def BlackHatFilter(img, filterSize =(3, 3)):
    img = self.rgb2gray(img)
    # Getting the kernel to be used in Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)
    # Applying the Top-Hat operation
    blackHatImg = cv2.morphologyEx(img,
                                  cv2.MORPH_BLACKHAT,
                                  kernel)
    return blackHatImg

def bilateralFilter(img, bilateralFilter =  [10, 10, 10]):
    return cv2.bilateralFilter(img, 25, 25, 25)


def cannyFilter(img, gausFil, cannyFil, kernelParam, dilateParam, erodeParam):
    imgGray = img.copy()#rgb2gray(img)

    imgBlur = bilateralFilter(img = imgGray)
    #imgBlur = cv2.GaussianBlur(imgGray, gausFil, 1) # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur,cannyFil[0],cannyFil[1]) # APPLY CANNY
    return imgCanny

    imgDial = dilate(imgCanny, kernelParam, dilateParam)
    imgCanny = erode(imgDial, kernelParam, erodeParam)
    return imgCanny


def maskHsv(img, down = (0, 0, 0), up= (0, 0, 255)):
    hsv_img = rgb2hsv(bgr2rgb(img))
    mask = cv2.inRange(hsv_img, down, up)
    result = cv2.bitwise_or(img, img, mask = mask)
    return result

# from pgmagick import Image

def sharpen(img,val):
    img.sharpen(val)
    return img




def gammaCalc(src, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT((255*src).astype(np.uint8), lookUpTable)
    return res.astype(np.float)/255

def gamma_correction(image: np.array, gamma: float = 1.4) -> np.array:
    """
    Applying gamma correction to image
    :param image: Image in float format
    :param size: Kernel size of Gaussian Blur filter
    :param sigma: Standard deviation of Gaussian Blur filter
    :param weight: Weight of unsharp masking
    :return: Image with applied unsharp mask
    """
    assert image.dtype == 'float64' or image.dtype == 'float32', \
    'Input image should be in float format. Given image in {} format'.format(image.dtype)
    ret = gammaCalc(image, gamma)#### PLACE YOUR CODE HERE ####
    return float2byte(ret)


def byte2float(image: np.array) -> np.array:
    """
    Converting image from byte to float representation
    :param image: Image in byte format
    :return: Image in float format
    """
    assert image.dtype == 'uint8', \
    'Input image should be in byte format. Given image in {} format'.format(image.dtype)
    return image.astype(float)/255

def float2byte(image: np.array) -> np.array:
    """
    Converting image from float to byte representation
    :param image: Image in float format
    :return: Image in byte format
    """
    assert image.dtype == 'float64' or image.dtype == 'float32', \
    'Input image should be in float format. Given image in {} format'.format(image.dtype)
    return (image*255).astype('uint8')




def increaseContrast(img,thre):
    img = Image.fromarray(img)
    width = img.size[0]
    height = img.size[1]
    for i in range(0,width):# process all pixels
        for j in range(0,height):
            data = img.getpixel((i,j))
            #print(data) #(255, 255, 255)
            if (data[0]<=thre[0] and data[1]<=thre[1] and data[2]<=thre[2]):
                img.putpixel((i,j),(0,0,0))
    return np.asarray(img)

def changeContarst(img, brightness=255,
               contrast=127):

    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified text string in the image.
#     cv2.putText(cal, 'B:{},C:{}'.format(brightness,
#                                         contrast), (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cal

    #----------------------------------------------------------------------------------------------------------------


def unsharp_mask(image: np.array,size: int = 9, sigma: float = 10,weight: float = 0.5) -> np.array:
    """
    Applying unsharp mask to image
    :param image: Image in float format
    :param size: Kernel size of Gaussian Blur filter
    :param sigma: Standard deviation of Gaussian Blur filter
    :param weight: Weight of unsharp masking
    :return: Image with applied unsharp mask
    """
    assert image.dtype == 'float64' or image.dtype == 'float32', \
    'Input image should be in float format. Given image in {} format'.format(image.dtype)
    #gaussian_rgb = cv2.GaussianBlur(image, (9,9), 10.0)
    gaussian_rgb = cv2.GaussianBlur(image, (size,size), sigma)#### PLACE YOUR CODE HERE ####
    unsharp_rgb = image + (image - gaussian_rgb)*weight
    # unsharp_rgb = cv2.addWeighted(image, 1-weight, gaussian_rgb, -1., weight)
    unsharp_rgb[unsharp_rgb < 0] = 0
    unsharp_rgb[unsharp_rgb > 1] = 1
    return float2byte(unsharp_rgb)
