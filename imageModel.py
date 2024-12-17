import tensorflow as tf
from keras_preprocessing import image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.morphology import dilation, disk

# ************************ MODELS ******************************

# Trained on 100K training dataset only
model1 = tf.keras.models.load_model('models/image/deepfake_detection_xception2.h5')  
# Accuracy - 70%

# Used model1 weights initially and then trained on 140K training dataset
model2 = tf.keras.models.load_model('models/image/deepfake_detection_xception3.h5')
# Accuracy - 76.23%

# Trained on 140K training dataset only
model3 = tf.keras.models.load_model('models/image/deepfake_detection_xception4.h5')
# Accuracy - 76.30%

# Trained on 180K training dataset with 10 epochs
model4 = tf.keras.models.load_model('best-image-model/180K-dataset/deepfake_detection_xception_180k.h5')

# Trained on 180K training dataset with 14 epochs 
model5 = tf.keras.models.load_model('best-image-model/180K-dataset/deepfake_detection_xception_180k_14epochs.h5')

# Mukul's models

model6 = tf.keras.models.load_model('./models/Image_Models_Mukul/DenseNet121Model.keras')
model7 = tf.keras.models.load_model('./models/Image_Models_Mukul/InceptionV3Model.keras')
model8 = tf.keras.models.load_model('./models/Image_Models_Mukul/ResNet50Model.keras')
model9 = tf.keras.models.load_model('./models/Image_Models_Mukul/XceptionModel.keras')
# ************************ MODELS ******************************

# def preprocess_image_predict(img_path):
#     # Preprocess the image
#     img = image.load_img(img_path, target_size=(256, 256))
#     img_array = image.img_to_array(img) / 255.0  # Normalize image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

def preprocess_image_predict(img_path, target_size=(256, 256)):
    # Load the image
    img = cv2.imread(img_path)
    
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale (required by the face detector)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If faces are detected, crop the first one (you can modify this to handle multiple faces)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]  # Crop the image to the face region
        
    # Resize the cropped face to the required size for the model (256x256 in this case)
    img = cv2.resize(img, target_size)
    
    # Convert the image to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image and prepare it for prediction
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def get_voted_prediction(img_path):
    # Preprocess the image for prediction
    img_array = preprocess_image_predict(img_path)

    # Get predictions from each model
    prediction1 = model1.predict(img_array)[0][0] 
    prediction2 = model2.predict(img_array)[0][0]
    prediction3 = model3.predict(img_array)[0][0]
    prediction4 = model4.predict(img_array)[0][0]
    prediction5 = model5.predict(img_array)[0][0]

    #Mukul's models predictions
    img_array = preprocess_image_predict(img_path,target_size=(224,224))
    prediction6 = model6.predict(img_array)
    prediction7 = model7.predict(img_array)
    prediction8 = model8.predict(img_array)
    prediction9 = model9.predict(img_array)
    
    print("Prediction1: ", prediction1)
    print("Prediction2: ", prediction2)
    print("Prediction3: ", prediction3)
    print("Prediction4: ", prediction4)
    print("Prediction5: ", prediction5)

    #Mukul's models predictions scores
    print("Prediction6: ", prediction6)
    print("Prediction7: ", prediction7)
    print("Prediction8: ", prediction8)
    print("Prediction9: ", prediction9)


    # Average the predictions for final score
    average_prediction = (prediction1 + prediction2 +  prediction3 +  prediction4 + prediction5) / 5
        
    score_real = round(float(average_prediction * 100), 2)
    score_fake = round(100 - score_real, 2)

    # Determine final class based on the average prediction
    predicted_class = 'Real' if average_prediction > 0.5 else 'Fake'
    
    return predicted_class, score_real, score_fake

def get_voted_prediction_video(img_path):
    # Preprocess the image for prediction
    img_array = preprocess_image_predict(img_path)

    # Get predictions from each model
    prediction = model5.predict(img_array)[0][0]
    
    print("Prediction: ", prediction)
        
    score_real = round(float(prediction * 100), 2)
    score_fake = round(100 - score_real, 2)

    # Determine final class based on the average prediction
    predicted_class = 'real' if prediction > 0.4 else 'fake'
    
    return predicted_class, score_real, score_fake

def explain_prediction_with_lime(img_path):
    # Preprocess the image for LIME
    img_array = preprocess_image_predict(img_path)
    
    # Create a LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Explain the prediction for the image using LIME
    explanation = explainer.explain_instance(
        img_array[0],  # Pass the first (and only) image in the batch
        model5.predict,  # Model's prediction function
        top_labels=1,  # Explain the top predicted label
        hide_color=0,  # Background color for unimportant regions
        num_samples=3000  # Number of perturbed samples for LIME
    )
    
    # Extract the explanation and mask for visualization
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Focus on the top predicted label
        positive_only=False,  # Include both positive and negative contributions
        num_features=10,  # Number of superpixels to highlight
        hide_rest=False  # Keep the entire image visible
    )
    
    # Dilate the mask to create bold boundaries
    dilated_mask = dilation(mask, disk(2))  # Adjust the radius for boldness

    # Apply the boundaries to the original image
    img_with_boundaries = mark_boundaries(
        (img_array[0] * 255).astype(np.uint8) / 255.0, 
        dilated_mask, 
        color=(1, 1, 0)  # Use black for boundaries
    )

    # Display the image with green and red contributions
    plt.figure(figsize=(6, 6))
    plt.title("LIME Explanation with Green and Red Contributions")
    plt.imshow(temp)  # Display the image with green (positive) and red (negative) regions
    plt.savefig("lime_explanation_green_red.png")  # Save for frontend
    # plt.close()

    # Display the image with overlaid yellow boundaries
    plt.figure(figsize=(6, 6))
    plt.title("LIME Explanation with Bold Yellow Boundaries")
    plt.imshow(img_with_boundaries)
    plt.axis('off')
    plt.savefig("lime_explanation_boundaries.png")  # Save for frontend
    # plt.close()

    # return "lime_explanation_green_red.png", "lime_explanation_boundaries.png"
