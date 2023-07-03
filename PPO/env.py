import numpy as np
import gym
gym.logger.set_level(40)
from gym import Env
from gym.spaces import Box, Discrete
import random
import cv2
import os
import tensorflow as tf
import logging
import torch
import torchvision.transforms as transforms
import os
from model50 import build_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Point(object):
    def __init__(self, x_max, x_min, y_max, y_min):
        self.x = 0 
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)      
#----------------------------------------------------------
class Roi(Point):

    def __init__(self, x_max, x_min,y_max,y_min):
        super(Roi, self).__init__(x_max,x_min,y_max,y_min)
        self.icon_w = 270
        self.icon_h = 270
        
#----------------------------------------------------------
class ImageEnv(Env):
    def __init__(self):
        super(ImageEnv, self).__init__()
        #Actions we can take such as: down, up, right, left
        self.action_space = Discrete(4)
        #Image shape
        self.width = 1080
        self.height = 1080
        #Define a 2-D observation space
        self.observation_shape = (self.width,self.height,1)
        # low = np.zeros((self.observation_shape))
        # high = np.ones(self.observation_shape) 
        self.observation_space = Box(low = 0, high = 255, shape= (360,1080,1), dtype = np.uint8)
        # Permissible area of ROI to be 
        self.y_min = 0
        self.x_min = 0
        self.y_max = self.observation_shape[1]
        self.x_max = self.observation_shape[0]
        # Position of ROI
        self.posx = 0
        self.posy = 0
        #Episode length 
        self.episodes = 300
        self.episode_length = self.episodes
        self.canvas = np.ones((1080,1080,3))
        self.roicell = np.ones((270,270,3)) 
        self.num_cells = 0
        self.getpos = []
        self.center = []
        self.count = 0 
        self.input_images_path = "/home/carlos/Documents/Decimo/Titulacion_II/Algorithms/base"
        self.model_cell = tf.keras.models.load_model("/home/carlos/Documents/Decimo/Titulacion_II/Algorithms/PPO/Cells_colab0.992_epo56")
        self.model_classify = torch.load('/home/carlos/Documents/Decimo/Titulacion_II/Algorithms/PPO/retraining/model.pth', map_location='cpu')
        self.history = []
        for i in range(0, 6):
            self.history.append(np.zeros((360, 360)))
        

    def draw_roi (self):
        #Init the canvas
        self.canvas = self.train/255.0 #---------------------------------------------
        innerix = self.roi.icon_h/4 + self.posx
        inneriy = self.roi.icon_w/4 + self.posy
        innerex = self.roi.icon_h/4 *3 + self.posx
        innerey = self.roi.icon_w/4 *3 + self.posy
        #Draw the ROI on canvas
        self.interest = self.image[self.posy:self.posy+self.roi.icon_w,self.posx:self.posx+self.roi.icon_h]
        cv2.rectangle(self.canvas,(self.posx,self.posy),(self.posx+self.roi.icon_h,self.posy+self.roi.icon_w),(0,0,0),3)
        cv2.rectangle(self.canvas,(int(innerix),int(inneriy)),(int(innerex),int(innerey)),(0,0,0),3)
        return self.interest


    
    def draw_nuclei(self,label):
        #Init the canvas
        self.user_image =self.train /255.0
        innerix = self.roi.icon_h/4 + self.posx
        inneriy = self.roi.icon_w/4 + self.posy
        innerex = self.roi.icon_h/4 *3 + self.posx
        innerey = self.roi.icon_w/4 *3 + self.posy
        #Draw the ROI in the nuclei
        self.roicell = self.image[self.posy:self.posy+self.roi.icon_w,self.posx:self.posx+self.roi.icon_h]
        cv2.putText(self.user_image,  f"{label}",(10,1060), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 155),3)

        if label == 'High squamous intra-epithelial lesion':
            cv2.rectangle(self.user_image,(self.posx,self.posy),(self.posx+self.roi.icon_h,self.posy+self.roi.icon_w),(0,0,1),3)
            cv2.rectangle(self.user_image,(int(innerix),int(inneriy)),(int(innerex),int(innerey)),(0,0,1),3)
        elif label == 'Low squamous intra-epithelial lesion' :
            cv2.rectangle(self.user_image,(self.posx,self.posy),(self.posx+self.roi.icon_h,self.posy+self.roi.icon_w),(0,1,0),3)
            cv2.rectangle(self.user_image,(int(innerix),int(inneriy)),(int(innerex),int(innerey)),(0,1,0),3)
        elif label == 'Negative for intra-epithelial malignancy':
            cv2.rectangle(self.user_image,(self.posx,self.posy),(self.posx+self.roi.icon_h,self.posy+self.roi.icon_w),(1,0,0),3)
            cv2.rectangle(self.user_image,(int(innerix),int(inneriy)),(int(innerex),int(innerey)),(1,0,0),3)
        else:
            cv2.rectangle(self.user_image,(self.posx,self.posy),(self.posx+self.roi.icon_h,self.posy+self.roi.icon_w),(0.6,0.6,0),3)
            cv2.rectangle(self.user_image,(int(innerix),int(inneriy)),(int(innerex),int(innerey)),(0.6,0.6,0),3)

        return self.roicell
    
    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up"}

    def step(self, action):
        image = self.pre_processing(self.canvas*255)
        # Draw elements on the canvas 
        self.draw_roi()
        self.center.append(self.posx)
        self.center.append(self.posy)
        # Flag that marks the termination of an episode
        done = False
        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"
        #Episode length 
        self.episode_length -= 1
        #ROI get the reward 
        self.posx, self.posy = self.roi.get_position()
        
        reward = -0.5
        if self.episode_length %2 == 0:
            accurrancy, label = self.cells(self.interest)
            if label == "Cell" and accurrancy>99:
                print(accurrancy)
                print(label)
                if self.posx in self.getpos and self.posy in self.getpos:
                    print("Already Stored")
                    reward = -10
                                        
                else:
                    label = self.classify(self.roicell)
                    self.num_cells += 1
                    self.getpos.append(self.posx)
                    self.getpos.append(self.posy)
                    self.draw_nuclei(label)
                    reward = 10 * self.num_cells

        if self.episode_length <= 0:
            done = True
        else: 
            done = False
        # apply the action to the ROI 
        if action == 0:
            self.roi.move(0,25)
        elif action == 1:
            self.roi.move(0,-25)
        elif action == 2:
            self.roi.move(25,0)
        elif action == 3:
            self.roi.move(-25,0)
        elif action == 4:
            self.roi.move(0,0)  
        #Set placeholder for info
        info = {}
        return image,reward, done,info
    
    def reset(self):
        print("Reseting the environment") 
        self.roicell = np.ones((270,270,3)) 
        self.num_cells = 0
        # Determine a place to intialise the ROI  in
        x = 0
        y = 0
        #Obtain the image values 
        self.image , self.train , self.reward_image, _= self.nenvironment(self.input_images_path)
        #Initialise the ROI
        self.roi = Roi(self.x_max, self.x_min, self.y_max, self.y_min)
        self.roi.set_position(x,y)
        #Restart the episode length
        self.episode_length = self.episodes
        #Create a canvas to render the environment images upon
        self.canvas = self.image
        # Reset the Canvas 
        image = self.pre_processing(self.canvas)
        #Reset the positions
        self.getpos = []
        self.center = []
        # Return the observation
        return image
        
    def nenvironment(self, path):
        
        files_names = os.listdir(path)
        file_name = files_names[random.randrange(0, len(files_names))]
        # file_name = files_names[self.count]
        image_path = path + "/" + file_name
        train = cv2.resize(cv2.imread(image_path), (1080, 1080))
        imagen = train.copy()
        self.reward_image = train.copy()
        self.user_image = train.copy()
        self.count += 1
        return imagen, train, self.reward_image, self.user_image


    def pre_processing(self, image):
        self.imagepre = cv2.cvtColor(cv2.resize(image.astype('uint8'), (360,360)), cv2.COLOR_BGR2GRAY)
        #_, self.imagepre = cv2.threshold(self.imagepre, 250, 255, cv2.THRESH_BINARY_INV)
        del self.history[0]
        self.history.append(self.imagepre)
        self.imagepre = np.concatenate((self.history[-5], self.history[-3],self.imagepre), axis=1)
        self.imagepre = np.expand_dims(self.imagepre, axis=-1)
        return self.imagepre.astype('uint8')
        
    def cells(self,img):
        Label = ["Cell", "None"]
        resize = cv2.resize(img, (50,50))
        normalize = resize /255.0
        reshaped = np.reshape(normalize,(1,50,50,3))
        predictions = self.model_cell.predict(reshaped)
        acurrancy = np.max(predictions)*100
        label = Label[np.argmax(predictions)]
        return acurrancy, label
    
    def classify(self, image):
        device = 'cpu'
        labels = ["High squamous intra-epithelial lesion", "Low squamous intra-epithelial lesion", "Negative for intra-epithelial malignancy", "Squamous cell carcinoma"]
        # initialize the model and load the trained weights
        model = build_model(
            pretrained=False, fine_tune=False, num_classes=4
        ).to(device)
        
        model.load_state_dict(self.model_classify['model_state_dict'])
        model.eval()
        # define preprocess transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,244)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) 
        # convert to RGB format
        img = cv2.convertScaleAbs(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        # add batch dimension
        img = torch.unsqueeze(img, 0)
        with torch.no_grad():
            outputs = model(img.to(device))
        output_label = torch.topk(outputs, 1)
        pred_class = labels[int(output_label.indices)]
        print(pred_class)
        return pred_class

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            winname = "Canvas"
            winname2 = "ROI"
            winname3 = "CNN"
            winname4 = "REWARD"
            winname5 = "User_image"
            cv2.namedWindow(winname)        # Create a named window
            cv2.moveWindow(winname, 500,20)  # Move it to (500,225)
            cv2.imshow(winname, cv2.resize(self.canvas, (400,400)))
            cv2.namedWindow(winname2)        # Create a named window
            cv2.moveWindow(winname2, 940,20)  # Move it to (930,225)
            cv2.imshow(winname2,self.interest)
            # cv2.namedWindow(winname3) 
            # cv2.moveWindow(winname3, 500,200)
            # cv2.imshow(winname3, self.pre_processing(self.canvas*255))
            cv2.namedWindow(winname4)
            cv2.moveWindow(winname4, 940, 500)
            cv2.imshow(winname4, self.roicell)
            cv2.namedWindow(winname5)
            cv2.moveWindow(winname5,500,500)
            cv2.imshow(winname5, cv2.resize(self.user_image, (400,400)))
            cv2.waitKey(1)   
        elif mode == "rgb_array":
            return
    def close(self):
        cv2.destroyAllWindows()
