from app.PlateDetection.commonfunctions import *
from skimage.feature import hog
from skimage.morphology import skeletonize
'''
There are 3 types of features
1-Structural features:will be number of dots,number of end points,number of loops,
2-Statistical features: will be number of connected components,zoning features
3-Global Transoformation :DCT,HOG
'''

# Range of dots is from 500 to 600 based on structural trials.
def DotsNum(char):
    Dots=0
    contours=cv2.findContours(char,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    maxArea=600
    minArea=500
    for contour in contours:
        if minArea<cv2.contourArea(contour)<maxArea:
            Dots+=1
    #show_images([char],["dots"])
    #print("No of dots: ",numDots)
    return Dots

# Get contours closed shape
def closedContours(char):
    contours,_=cv2.findContours(char, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def EndPoints(char):
    if np.max(char)>1: 
        char=np.where(char==255,1,0)

    skel=skeletonize(char)
    
    
    skel = np.uint8(skel)
    kernel = np.uint8([[1,  1, 1],
                       [1, 50, 1],
                       [1,  1, 1]
                       ])
    filtered = cv2.filter2D(skel,-1,kernel)
    out = np.zeros_like(skel)
    out[np.where(filtered==51)] = 1
    count=np.unique(out,return_counts=True)[1]
    if len(count)>1:
        #print(count[1])
        return count[1] 
    return 0

def ConnectedComponents(char):
    output = cv2.connectedComponentsWithStats(char, 8, cv2.CV_32S)
    return output[0]

def DiagonalFeatures(char):
    firstQ=char[0:len(char[0])//2][0:len(char[1])//2].diagonal()
    secondQ=char[len(char[0])//2:len(char[0])][0:len(char[1])//2].diagonal()
    thirdQ=char[0:len(char[0])//2][len(char[1])//2:len(char[1])].diagonal()
    fourthQ=char[len(char[0])//2:len(char[0])][len(char[1])//2:len(char[1])].diagonal()
    diagonalSum=[np.sum(firstQ)/255,np.sum(secondQ)/255,np.sum(thirdQ)/255,np.sum(fourthQ)/255]
    return diagonalSum

def VerticalFeatures(char):
    blocks=[]
    offsetX=len(char)//16
    currentX=0
    
    for _ in range (0,15):
        blocks.append(char[currentX:currentX+offsetX][:])
        currentX+=offsetX    
    blocksSum=[]
    for block in blocks:
        blocksSum.append(np.sum(block)/255)
    return blocksSum

def HorizontalFeatures(char):
    blocks=[]
    offsetX=len(char)//16
    currentX=0
    for _ in range (0,15):
        blocks.append(char[:][currentX:currentX+offsetX])
        currentX+=offsetX    
    blocksSum=[]
    for block in blocks:
        blocksSum.append(np.sum(block)/255)
    return blocksSum

def Dct(char):
    imageFloat=np.float32(char)/255.0  
    imageFloat=cv2.resize(imageFloat,(200,200))
    dct = cv2.dct(imageFloat)           # the dct
    dctCo = np.uint8(dct)*255.0    # convert back
    dctV=[[] for i in range(len(dctCo[1])+len(dctCo[0])-1)]
    for i in range(0,len(dctCo[0])): 
        for j in range(0,len(dctCo[1])): 
            sum=i+j 
            if(sum%2 ==0): 
                #add at beginning 
                dctCo[i][j]
                dctV[sum].insert(0,dctCo[i][j]) 
            else: 
                #add at end of the list 
                dctV[sum].append(dctCo[i][j]) 
    dctV = [item for sublist in dctV for item in sublist]
    return dctV[0:200] 

def getHogFeatures(char):
    hogComputer=cv2.HOGDescriptor((64,64),(32,32),(16,16),(16,16),9)
    x=hogComputer.compute(cv2.Canny(cv2.resize(char,(64,64)),0,255))
    return x

def extractFeatures(char):
    ALLfeature=[DotsNum(char),closedContours(char),EndPoints(char),ConnectedComponents(char)]
    ALLfeature.extend(DiagonalFeatures(char))
    ALLfeature.extend(HorizontalFeatures(char))
    ALLfeature.extend(VerticalFeatures(char))
    ALLfeature.extend(Dct(char))
    ALLfeature.extend(getHogFeatures(char).flatten())
    
    #Normalizing vector
    maxFeatureValue=max(ALLfeature)
    minFeatureValue=min(ALLfeature)
    for i in range (0,len(ALLfeature)):
        ALLfeature[i]=(ALLfeature[i]-minFeatureValue)/(maxFeatureValue-minFeatureValue)
    return ALLfeature 



def PreProcessingImage(img):
    # (1) RGB to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (2) threshold
    threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return threshed

def extract_sift(img):
    # create SIFT feature extractor
    #img = cv.resize(img, (128, 128))
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descrip = sift.detectAndCompute(img, None)
    
    
    descriptors = descrip[0]
    for descriptor in descrip[1:]:
        np.append(descriptors, descriptor) 

    return descriptors[:200]


def extract_hog(img):
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 200))
    fd = hog(img, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1))
    #hog = cv.HOGDescriptor()
    #hog_descriptor = hog.compute(img)
    return fd

