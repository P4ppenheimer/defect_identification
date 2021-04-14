# general stuff
import pandas as pd
import os 

# torch stuff
import torch
from torchvision.transforms import Resize, Scale, Normalize, RandomHorizontalFlip, ToTensor, Compose, ToPILImage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom functions
from helper_functions import *

# constants
best_threshold = 0.5
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_pivot_table(df):
    
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    return df 

def get_prediction(idx, model):
    
    df_path = 'input/train.csv'

    df = pd.read_csv(df_path)
    df = get_pivot_table(df)

    image_id, mask = make_mask(idx, df)
    print('image_id', image_id)

    image_path = os.path.join('input', "train_images",  image_id)
    img = cv2.imread(image_path)

    list_trfms = Compose([ToTensor(),Normalize(mean=mean, std=std),])
    img = list_trfms(img)

    img = img.unsqueeze(0)


    preds = torch.sigmoid(model(img.to(device)))
    preds = preds.detach().cpu().numpy()  # (1 x 4 x h x w)

    preds = preds.squeeze(0)

    pred_mask = None
    pred_class = None
    pred_num = None 

    for class_, pred_per_class in enumerate(preds):

        pred, num = post_process(pred_per_class, best_threshold, min_size)

        print(f'class_ {class_} and num {num}')
        if num > 0: # not sure but I think num means number of connected segmentations
            pred_mask = pred
            pred_class = class_
            pred_num  = num

    preds = torch.sigmoid(model(img.to(device)))
    preds = preds.detach().cpu().numpy()  # (1 x 4 x h x w)

    masks = torch.tensor(mask).permute(2,0,1) # 4x256x1600

    img = img.squeeze(0)

    img = inverse_normalize(tensor=img, 
                            mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))

    img = img.permute(1, 2, 0).numpy() # 

    # masks 4 x 400 x 1600 - one mask for each class
    for mask_idx, mask in enumerate(masks):

        # if mask does not consist of only zeros, then this is the mask of the true label
        if not (mask == 0).all():

            true_mask = mask.numpy()
            true_class = mask_idx + 1 # because we have classes 1, .. , 4 which are encoded by 0, .., 3 
    
    img[true_mask==1,0] = 1 # we set the first (=red) channel dim to 255 for true mask
    img[pred_mask==1,1] = 1 # we set the second (=green) channel dim to 255 for pred mask

    # channels RGB = red, green, blue


    # plt.figure(figsize=(20,100))

    # plt.gca().set_title(f'Image ID {idx} with image name {image_id} \n Pred. class {pred_class} and true class {true_class} and num {pred_num} \n red pixels == true mask ; green pixels == pred mask ; yellow pixels == coincidence of predicted and true mask')
    # plt.imshow(img)
    # plt.show()
    
    return img