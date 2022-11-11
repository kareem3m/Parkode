import cv2 as cv
import numpy as np
from app.PlateDetection.commonfunctions import *
from numba import jit
import pickle
from app.PlateDetection.featureExtraction import *

# load saved model
letters = {1:'ع', 2:'أ', 3:'ب', 4:'د', 5:'ف', 6:'ج', 7:'ه', 8:'ل', 9:'م', 10:'ن', 11:'ق', 12:'ر', 13:'ص', 14:'س', 15:'ط', 16:'و', 17:'ى'}
print("CWD=========", os.getcwd())
# letter_model = pickle.load(open('./app/PlateDetection/RF_model_hog_L.sav', 'rb'))
# number_model = pickle.load(open('./app/PlateDetection/RF_model_hog_N.sav', 'rb'))
letter_model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'RF_model_hog_L.sav'), 'rb'))
number_model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'RF_model_hog_N.sav'), 'rb'))

# flag = 0 --> digits classification|| flag = 1 --> letters classification
def i2ocr(img, flag):
    if flag:
        # Preprocess image...
        img_processed = PreProcessingImage(img)
        # Feature Extraction...
        hog_feature = extractFeatures(img_processed)
        test_img_feature = []
        Features = np.concatenate([hog_feature])
        test_img_feature.append(Features)
        # Prediction...
        label_L = letter_model.predict(test_img_feature)
        return letters[int(label_L)]
    else:
        # Preprocess image...
        img_processed = PreProcessingImage(img)
        # Feature Extraction...
        hog_feature = extract_hog(img_processed)
        test_img_feature = []
        Features = np.concatenate([hog_feature])
        test_img_feature.append(Features)
        # Prediction...
        label_L = number_model.predict(test_img_feature)
        return str(int(label_L))


@jit(nopython=True)
def WB_region(image, aux):
    h = image.shape[0]
    w = image.shape[1]
    white = 200
    local_thre = 30
    global_thre = 60

    for y in range(0, h):
        for x in range(0, w):
            b,g,r = image[y,x]
        
            s,m,l = np.sort(image[y,x])
            
            local_dis = (m-l)*(m-l)+(m-s)*(m-s)
            aux[y, x, 0] = 1 if (local_dis<local_thre*local_thre and abs(white-(s+m+l)/3)<global_thre) else 0
            aux[y, x, 1] = 1 if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110) else 0

def check_blue(pixel):
        b,g,r = pixel
        s,m,l = np.sort(pixel)
        if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110):
            return True
        else:
            return False

def OutRange(aux, Xmin, Ymin, Xmax, Ymax): 
    res = aux[Ymax][Xmax]
    if (Ymin > 0): 
        res = res - aux[Ymin - 1][Xmax] 
        
    if (Xmin > 0):
        res = res - aux[Ymax][Xmin - 1] 
    
    if (Ymin > 0 and Xmin > 0): 
        res = res + aux[Ymin - 1][Xmin - 1] 
    return res 

def Bluring(img):
    blur = cv.GaussianBlur(img, (3, 3), 0)
    return blur


def Thresholding(gray_img):
    bin_img = cv.threshold(gray_img,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    return bin_img

def detect_edges(gray_img):
    #Horizontal Edge detection
    x = cv.Sobel(gray_img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    #Vertical Edge detection 
    y = cv.Sobel(gray_img, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    x = cv.convertScaleAbs(x)
    y = cv.convertScaleAbs(y)
    return cv.addWeighted(x, 0.5, y, 0.5, 0)

#Can be change according to the setup of the camera (min and max aspect ratio - if it is far or not) 
def plate_criteria(cum_white, cum_blue, x, y, w, h, aspect_min, aspect_max, far): 
    area = w*h
    [Xmin,Ymin,Xmax,Ymax] = [x,y,x+w-1,y+h-1]
    if(h>0 and aspect_min < float(w)/h and float(w)/h < aspect_max): #Check Aspect ratio
        if(area >= cum_white.shape[0] * cum_white.shape[1] * far): #check far or not
            white_ratio = OutRange(cum_white, Xmin, Ymin, Xmax, Ymax)/area*100
            blue_ratio = OutRange(cum_blue, Xmin, Ymin, Xmax, Ymax)/area*100
            if(white_ratio > 35 and white_ratio < 90 and blue_ratio > 7 and blue_ratio < 40):
                return True
    return False

def ROI(img, bin_img, aspect_min, aspect_max, far):
    
    # Finding contours
    Contours= cv.findContours(bin_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    
    aux = np.copy(img)
    WB_region(img,aux)

    # Cumulative white and blue pixels to detect the plate structure in x
    whiteSum = np.cumsum(aux[:,:,0], axis = 0).astype(np.int64)
    blueSum = np.cumsum(aux[:,:,1], axis = 0).astype(np.int64)

    # Cumulative white and blue pixels to detect the plate structure in y
    whiteSum = np.cumsum(whiteSum, axis = 1).astype(np.int64)
    blueSum = np.cumsum(blueSum, axis = 1).astype(np.int64)

    for c in Contours:    
        [x,y,w,h] = cv.boundingRect(c)
        if(plate_criteria(whiteSum, blueSum, x, y, w, h,aspect_min, aspect_max, far)):
            # not out of range
            if(y-h/4>=0):
                return np.copy(img[y-int(h/4):y+h-1,x:x+w-1]),1
            else:    
                return np.copy(img[y:y+h-1,x:x+w-1]),1
    return img,0


def localizationCountour(img):
    # Preprocess image...
    img = Bluring(img)
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = detect_edges(gray_img)
    kernel = np.ones((5,5),np.uint8)
    morphoclose = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Binarize before contour detection
    img_th = Thresholding(morphoclose)
    # Find contours of the interested region
    plateArea,flag = ROI(img, img_th, 1.4, 2.5, 0.01) 

    
    # Binarize before contour detection due to plate region noise
    plate_area_img_bin = cv.adaptiveThreshold(cv.cvtColor(255-plateArea,cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,8)
    # Find the plate contour exactly
    plate_img = ROI(plateArea, plate_area_img_bin, 1, 2.1, 0.1)[0]
    if(flag):
        return crop_up(plate_img),flag
    return plate_img,flag

# Dilation and Erosion to remove some noise
def dilate(image):
    kernel = np.ones((7,7),np.uint8)
    return cv.dilate(image, kernel, iterations = 2)

def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv.erode(image, kernel, iterations = 2)

# Crop the plate (White region only)
def crop_up(img):
    y = img.shape[0]
    x = int (img.shape[0]/2)
    # Crop the plate region right after the blue region
    for i in range(0,y):
        if(check_blue(img[i][x])):
            return img[i:y,0:img.shape[1]]
    return img

def get_blue(frame):    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    lower_blue = np.array([30, 130, 100],np.uint8)
    upper_blue = np.array([128, 255, 255],np.uint8)
    
    # Get blue channel from the blue range
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # To connect the blue regions together
    mask = dilate(mask)

    #show_images([cv.cvtColor(mask, cv.COLOR_BGR2RGB)])

    # Percentage of blue in the frame
    if (cv.countNonZero(mask)/(frame.shape[0]*frame.shape[1])) < 0.05:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        c = max(contours, key = cv.contourArea)
        dim = cv.boundingRect(c)
        return dim
    return 0

def detect(img):
    ######################### Plate Detection ############################
    # Get the region of interest
    dimension = get_blue(img)
    if (dimension != 0):
        c, r, w, h = dimension[0], dimension[1], dimension[2], dimension[3]
        img = img[r-h:r+h*3,c-50:c+w+50]
    else:
        return "Plate not Found :("
    # Get the plate contour
    crop, f= localizationCountour(img)
    
    
    ########################## Segmentation ###############################
    original = crop.copy()

    # Apply binarization
    crop = Thresholding(cv.cvtColor(crop, cv.COLOR_BGR2GRAY))
    crop = erode(crop)
    crop = Bluring(crop)

    kernel = np.ones((5,5),np.uint8)
    crop = cv.morphologyEx(crop, cv.MORPH_CLOSE, kernel)
    crop = erode(crop)

    crop = Bluring(crop)
    crop = cv.dilate(crop, (7,7), iterations = 1)

    # Letters & Digits
    spliter = crop.shape[1]//2

    contours= cv.findContours(crop.copy(), cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

    res = ""
    for i,ctr in enumerate(contours,0):
        # Get bounding box
        x, y, w, h = cv.boundingRect(ctr)

        #Connected Component analysis
        MOW = crop.shape[1]//6
        MOH = int(0.6 * (crop.shape[1]/2.8))
        #print(MOW, MOH)
        RT = w/h
        # Getting ROI
        margin = 40
        #show_images([original[y:y+h, x:x+w]])
        roi = original[y-margin:y+h+margin, x-5:x+w+5]
        
        # Character Recognition
        if w <= MOW and h/crop.shape[0]>=0.1 and h <= MOH and roi is not None and w * h > 4000:
            #show_images([roi])
            if x <= spliter:
                res += i2ocr(roi,0)
            else:
                res += i2ocr(roi,1)
    return res


