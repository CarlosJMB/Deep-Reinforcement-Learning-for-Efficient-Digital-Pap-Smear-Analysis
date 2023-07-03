import torch
import cv2
import torchvision.transforms as transforms
import os
import random
from model50 import build_model


test_dir = '/home/carlos/Desktop/Decimo/Titulacion_II/FP_Categories_Classification/image.png'

# the computation device
device = 'cpu'
# list containing all the labels
labels = ["High squamous intra-epithelial lesion", "Low squamous intra-epithelial lesion", "Negative for intra-epithelial malignancy", "Squamous cell carcinoma"]

# initialize the model and load the trained weights
model = build_model(
    pretrained=False, fine_tune=False, num_classes=4
).to(device)
print('[INFO]: Loading custom-trained weights...')
checkpoint = torch.load('outputs50/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
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
while True:
    class_num = str(random.randrange(1,5))
    class_path = test_dir + "/" + class_num
    files_names = os.listdir(class_path)
    image_path = class_path + "/" + files_names[random.randrange(0,len(files_names))]

    # read and preprocess the image
    image = cv2.imread(image_path)
    # get the ground truth class
    if class_num == '1':
        gt_class = 'High squamous intra-epithelial lesion'
    elif class_num == '2':
        gt_class = 'Low squamous intra-epithelial lesion'
    elif class_num == '3':
        gt_class = 'Negative for intra-epithelial malignancy'
    else:
        gt_class = 'Squamous cell carcinoma'

    orig_image = image.copy()
    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    cv2.putText(orig_image, 
        f"GT: {gt_class}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 255, 0), 2
    )
    cv2.putText(orig_image, 
        f"Pred: {pred_class}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2
    )
    print(f"GT: {gt_class}, pred: {pred_class}")
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    # cv2.imwrite(f"outputs/{gt_class}.png",
    #     orig_image)
