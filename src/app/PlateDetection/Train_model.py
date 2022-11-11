#########################################################################
################################### OCR #################################
#########################################################################
from featureExtraction import *
from commonfunctions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle




def trainLetters():
    letters = ['3en','alf','bih','dal','feh','gem','heh','lam', 'mem', 'non', 'qaf', 'reh', 'sad', 'sen', 'tah', 'wow', 'yeh']
    xtrain = []
    for l in letters:
        xtrain.append(load_images_from_folder('Data_Set/Letters/'+l))
    xtrain = sum(xtrain, [])
    
    letter_labels = np.ones(len(xtrain))
    letter_labels[:21] = 1        # 3en
    letter_labels[21:43] = 2      # alf
    letter_labels[43:63] = 3      # bih 
    letter_labels[63:84] = 4      # dal
    letter_labels[84:114] = 5     # feh
    letter_labels[114:137] = 6    # gem
    letter_labels[137:164] = 7    # heh
    letter_labels[164:192] = 8    # lam
    letter_labels[192:209] = 9    # mem
    letter_labels[209:240] = 10   # non
    letter_labels[240:267] = 11   # qaf
    letter_labels[267:286] = 12   # reh
    letter_labels[286:301] = 13   # sad
    letter_labels[301:319] = 14   # sen
    letter_labels[319:339] = 15   # tah
    letter_labels[339:365] = 16   # wow
    letter_labels[365:377] = 17   # yeh
    
    return xtrain, letter_labels
    
def trainNumbers():
    numbers = ['1','2','3','4','5','6','7','8','9']
    xtrain = []
    for n in  numbers:
        xtrain.append(load_images_from_folder('Data_Set/Numbers/'+n))
    xtrain = sum(xtrain, [])
    
    number_labels = np.ones(len(xtrain))
    number_labels[:43] = 1        # 1
    number_labels[43:89] = 2      # 2
    number_labels[89:155] = 3     # 3 
    number_labels[155:201] = 4    # 4
    number_labels[201:267] = 5    # 5
    number_labels[267:322] = 6    # 6
    number_labels[322:364] = 7    # 7
    number_labels[364:418] = 8    # 8
    number_labels[418:468] = 9    # 9

    return xtrain, number_labels




xLetter, y_letter = trainLetters()

xNumber, y_number = trainNumbers()


X_letter_FE = []
for i in xLetter:
    i = PreProcessingImage(i)
    #f1 = lpq(i)
    f2 = extractFeatures(i)
    #f3 = extract_sift(i)
    #X_let = np.concatenate([f2])
    X_letter_FE.append(f2)


X_number_FE = []
for i in xNumber:
    i = PreProcessingImage(i)
    #f1 = lpq(i)
    f2 = extract_hog(i)
    #f3 = extract_sift(i)
    #X_num = np.concatenate([f1,f2])
    X_number_FE.append(f2)


########################## Letters ##################################
clf_RF_L = RandomForestClassifier(max_depth=5, random_state=42).fit(X_letter_FE, y_letter)
#clf_SVM_L = SVC(kernel='rbf' ,random_state=42, gamma=0.01).fit(X_letter_FE, y_letter)
# evaluate on the train dataset
clf_RF_L.predict(X_letter_FE)
#clf_SVM_L.predict(X_letter_FE)
# Accuracies
train_scores_L = []
train_scores_L.append(clf_RF_L.score(X_letter_FE, y_letter))
#train_scores_L.append(clf_SVM_L.score(X_letter_FE, y_letter))

########################## Numbers ##################################
clf_RF_N = RandomForestClassifier(max_depth=5, random_state=42).fit(X_number_FE, y_number)
#clf_SVM_N = SVC(kernel='rbf' ,random_state=5, gamma=0.01).fit(X_number_FE, y_number)
# evaluate on the train dataset
clf_RF_N.predict(X_number_FE)
#clf_SVM_N.predict(X_number_FE)
# Accuracies
train_scores_N = []
train_scores_N.append(clf_RF_N.score(X_number_FE, y_number))
#train_scores_N.append(clf_SVM_N.score(X_number_FE, y_number))



####################### Accuracies ###################
print(train_scores_L,train_scores_N)

################ Saving the Model ########################
# save the model
pickle.dump(clf_RF_L, open('RF_model_hog_L.sav', 'wb'))
pickle.dump(clf_RF_N, open('RF_model_hog_N.sav', 'wb'))