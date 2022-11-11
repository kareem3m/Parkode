from featureExtraction import *

def Car_count(DIR):
    count = 0
    for path in os.listdir(DIR):
        if os.path.isdir(os.path.join(DIR, path)):
            count += 1
    return count

def getMyCar():
    
    # read imgs from test file
    images_N = []
    text_N = []

    CarCount = Car_count('Work flow/Characters Detected/')
    for i in range(1,CarCount-1):
        images_N = load_images_from_folder("Work flow/Characters Detected/"+str(i)+"/Num/")
        result = ""
        for j in images_N:
            #show_images([j])
            # Preprocessing...
            test_img_processed = PreProcessingImage(j)

            # Feature Extraction...
            hog_feature = extract_hog(test_img_processed)
            test_img_feature = []
            Features = np.concatenate([hog_feature])
            test_img_feature.append(Features)

        
            label_L = number_model.predict(test_img_feature)
            
            result += str(int(label_L))
        text_N.append(result)

    x = input("Enter your license plate number: ")
    if x in text_N:
        print("Found it!")
    else:
        print("Not Found: your car was stolen!")