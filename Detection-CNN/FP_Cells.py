
#STUDENT: CARLOS JULIO MACANCELA BOJORQUE####################
#############Necessary libraries############
import numpy as np
import os
import PIL
import PIL.Image
from PIL import ImageFont
import pathlib
import matplotlib.pyplot as plt
import random
import cv2
############## Unnecessary Warning Messages###############
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf    #specialized libraries
from tensorflow import keras #specialized libraries
from keras import layers, models # Importing the pre designed layers of a NN
from keras.preprocessing import image  # Some extra modules for image preporcesing
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from plot_keras_history import show_history, plot_history
import visualkeras
##########################################################

batch_size = 32 #Defining the batch size 
img_height = 20 # The height image
img_width = 20  # The width image

DB_dir_train = pathlib.Path(r"/home/carlos/Desktop/Decimo/Titulacion_II/Algorithms/FP_Cells/DB_3/train") # get data base path
DB_dir_val = pathlib.Path(r"/home/carlos/Desktop/Decimo/Titulacion_II/Algorithms/FP_Cells/DB_3/validation") # get data base path


image_count_train = len(list(DB_dir_train.glob("*/*g"))) # get dataset size
image_count_val = len(list(DB_dir_val.glob("*/*g"))) # get dataset size

print("\nTraining Images Data Base Size: ", image_count_train, "\n")
print("Validation Images Data Base Size: ", image_count_val, "\n")
class_names = list(sorted([item.name for item in DB_dir_train.glob('*')]))  # Get the name of the classe in this case two [0, 1]
print(class_names, "\n")

datagen = ImageDataGenerator(rescale=1./255)
train = datagen.flow_from_directory(DB_dir_train, target_size=(img_height, img_width), shuffle=True, batch_size=batch_size, class_mode='sparse') # Get the images of the training folder
val = datagen.flow_from_directory(DB_dir_val, target_size=(img_height, img_width), shuffle=False, batch_size=batch_size, class_mode='sparse')# Get the images of the validation folder
print("\n")

#How many epochs the NN will train
e=50

model = models.Sequential()
model.add(layers.Conv2D(10, (10,10), activation='relu', input_shape=(img_height, img_width, 3))) 
model.add(layers.MaxPooling2D(pool_size=(2,2))) 
model.add(visualkeras.SpacingDummyLayer(spacing=25))
model.add(layers.Flatten()) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2, activation='softmax')) 
model.summary()  

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train, epochs=e) # Start the training during 100 epochs
# Get the loss and acurrancy of our NN using the validation data
print(history.history.keys())
val_loss, val_accuracy = model.evaluate(val)
print("Val_Loss: ",val_loss)
print("Val_Accuracy: ",val_accuracy)

# test_loss, test_accuracy = model.evaluate(test)
# print("Val_Loss: ",test_loss)
# print("Val_Accuracy: ",test_accuracy)
# plot_model(model,to_file= "model.png" )
font = ImageFont.truetype("lmroman12-regular.otf", 18)  
visualkeras.layered_view(model, legend =True, font=font, to_file = 'architecture_paper.png').show()
# visualkeras.layered_view(model,to_file='visualize.png').show()
model.save("Detection_Cells"+str(round(val_accuracy, 3))+"_epo_" + str(e)) #save model

show_history(history)
plot_history(history, path="/home/carlos/Desktop/Decimo/Titulacion_II/Algorithms/FP_Cells/accurancy_loss2.png")
plt.close()

#Label = ["Cell","No Cell"]
# print("Loading Model")
# model = tf.keras.models.load_model("/home/carlos/Desktop/Decimo/Titulacion_II/Algorithms/FP_Cells/FP_Cells_new0.961_epo_30")

# predictions = model.predict(val)
# l1 = list(range(len(val)-1))
# random.shuffle(l1)
# l2 = list(range(30))
# random.shuffle(l2)

# for i in l2:
#     for b in l1:
#         print("Label: " + str(val[b][1][i]))
#         print("Prediction: " + str(Label[np.argmax(predictions[i+b*batch_size])]) + " ------> accuracy: " + str(np.max(predictions[i+b*batch_size])*100))
#         plt.figure()
#         plt.imshow(val[b][0][i])
#         plt.title("Result \n Prediction: " + str(Label[np.argmax(predictions[i+b*batch_size])]) + " | " + str(round(np.max(predictions[i+b*batch_size])*100, 3)) + " % of acurracy \n\n" + "Label: " + str(Label[int(val[b][1][i])]))
#         plt.show()
