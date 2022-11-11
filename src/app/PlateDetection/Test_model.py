from featureExtraction import *
from skimage.feature import hog
import pickle
import cv2 as cv
from sklearn.metrics import fbeta_score, accuracy_score

########################## testing... ########################

# load saved model
letter_model = pickle.load(open('RF_model_hog_L.sav', 'rb'))
number_model = pickle.load(open('RF_model_hog_N.sav', 'rb'))

letters = ['3en','alf','bih','dal','feh','gem','heh','lam', 'mem', 'non', 'qaf', 'reh', 'sad', 'sen', 'tah', 'wow', 'yeh']
letters_label = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 
                 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                 13, 13, 13, 13, 14 ,14 ,14 ,14, 15, 15, 15 ,15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17]




# start time and open file outputs
#results_file = open(out_path +'/results.txt',"w")
#time_file = open(out_path +'/times.txt',"w")
labels_L = []
for i in letters:
    # read imgs from test file
    test_images_L = load_images_from_folder('TestSet/Alphabets/'+i)
    ########################### Letters ##########################
    for test_img in test_images_L:
        #start = time.time()
        # preprocessing for test file 
        test_img_processed = PreProcessingImage(test_img)
        
        #show_images([test_img_processed])
        
        # feature extraction
        test_img_feature = []
        
        hog_feature = extractFeatures(test_img_processed)
        Features = np.concatenate([hog_feature])
        test_img_feature.append(Features)
        
        # classification for the test file
        label_L = letter_model.predict(test_img_feature)
        #label1 = loaded_model1.predict(test_img_feature)
        # end time
        #end = time.time()

        # print outputs
        #Timer = end - start

        #check if predict function returns None
        if label_L is None:
            #results_file.write(str(-1))
            print("None")
        else:
            #results_file.write(str(int(label[0])))
            labels_L.append(int(label_L))
        
print("Letters Accuracy = {0}%".format(accuracy_score(letters_label, labels_L)*100))


Numbers_label = []
labels_N = []

########################### Numbers ##########################
# Main processing
for i in range(1,10):
    # read imgs from test file
    test_images_N = load_images_from_folder("TestSet/Numbers/"+str(i))
    for test_img in test_images_N:

        Numbers_label.append(i)
        #start = time.time()
        # preprocessing for test file 
        test_img_processed = PreProcessingImage(test_img)
        #show_images([test_img_processed])
        # feature extraction
        test_img_feature = []
        
        
        hog_feature = extract_hog(test_img_processed)
        Features = np.concatenate([hog_feature])
        test_img_feature.append(Features)
        
        
        # classification for the test file
        label_N = number_model.predict(test_img_feature)
        #label1 = loaded_model1.predict(test_img_feature)
        # end time
        #end = time.time()

        # print outputs
        #Timer = end - start

        #check if predict function returns None
        if label_N is None:
            #results_file.write(str(-1))
            print("None")
        else:
            #results_file.write(str(int(label[0])))
            labels_N.append(int(label_N))

print("Numbers Accuracy = {0}%".format(accuracy_score(Numbers_label, labels_N)*100))