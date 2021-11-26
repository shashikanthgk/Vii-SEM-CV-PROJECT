import torch
import segmentation_models_pytorch as smp
import csv
import dataset as ds
import preprocess as pp
from torch.utils.data import DataLoader
import loss as lossfile
import os
import numpy as np
import cv2
class Segmodel:
    def __init__(self,ENCODER,ENCODER_WEIGHTS,ACTIVATION):
        self.ENCODER = ENCODER
        self.ENCODER_WEIGHTS = ENCODER_WEIGHTS
        self.ACTIVATION = ACTIVATION
        self.base_model = None
        self.train_loader = None
        self.valid_loader = None
    def initialize_basemodel(self,base_model,class_len):
        self.base_model = base_model(encoder_name = self.ENCODER, 
                                          encoder_weights = self.ENCODER_WEIGHTS, 
                                          classes = class_len, 
                                          activation = self.ACTIVATION)

    def fit(self,x_train_dir, y_train_dir,x_valid_dir, y_valid_dir,select_class_rgb_values):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)
        train_dataset = ds.BuildingsDataset(
                                        x_train_dir, y_train_dir, 
                                        augmentation = ds.get_training_augmentation(),
                                        preprocessing = ds.get_preprocessing(preprocessing_fn),
                                        class_rgb_values = select_class_rgb_values,
                                        )

        valid_dataset = ds.BuildingsDataset(
                                        x_valid_dir, y_valid_dir, 
                                        augmentation = ds.get_validation_augmentation(), 
                                        preprocessing = ds.get_preprocessing(preprocessing_fn),
                                        class_rgb_values = select_class_rgb_values,
                                        )   

        self.train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 12)
        self.valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 4)


    def train(self,ephocs,loss_fun,metrics,optimizer,lr,lr_scheduler,model_name):
        if(self.valid_loader == None or self.train_loader == None):
            print("You have to fit the model before training !")
            return 
        TRAINING = True
        EPOCHS = ephocs
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optimizer([dict(params = self.base_model.parameters(), lr = lr)])
        train_epoch = smp.utils.train.TrainEpoch(
                                                    self.base_model, 
                                                    loss = loss_fun, 
                                                    metrics = metrics, 
                                                    optimizer = optimizer,
                                                    device = DEVICE,
                                                    verbose = True,
                                                )

        valid_epoch = smp.utils.train.ValidEpoch(
                                                    self.base_model, 
                                                    loss = loss_fun, 
                                                    metrics = metrics, 
                                                    device = DEVICE,
                                                    verbose = True,
                                                )
        if TRAINING:
            best_iou_score = 0.0
            train_logs_list, valid_logs_list = [], []

            for i in range(0, EPOCHS):
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(self.train_loader)
                valid_logs = valid_epoch.run(self.valid_loader)
                train_logs_list.append(train_logs)
                valid_logs_list.append(valid_logs)
                if best_iou_score < valid_logs['iou_score']:
                    best_iou_score = valid_logs['iou_score']
                    torch.save(self.base_model, model_name+".pth")
                    print('Model saved!')
        if ephocs != 0:
            labels = train_logs_list[0].keys()
            try:
                with open(model_name+"_train.csv", 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=labels)
                    writer.writeheader()
                    for elem in train_logs_list:
                        writer.writerow(elem)
            except IOError:
                    print("I/O error")
            try:
                with open(model_name+"_valid.csv", 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=labels)
                    writer.writeheader()
                    for elem in train_logs_list:
                        writer.writerow(elem)
            except IOError:
                    print("I/O error")

def train_loss2(lambda_,model_unet,CLASSES,x_train_dir, y_train_dir,x_valid_dir, y_valid_dir,select_class_rgb_values):
    model_unet.initialize_basemodel(smp.Unet,len(CLASSES))
    model_unet.fit(x_train_dir, y_train_dir,x_valid_dir, y_valid_dir,select_class_rgb_values)
    EPOCHS = 50
    loss = lossfile.CustomLoss2(lambda_)
    metrics = [
        smp.utils.metrics.IoU(threshold = 0.5),
    ]
    optimizer = torch.optim.Adam
    lr = 0.0001
    model_unet.train(EPOCHS,loss,metrics,optimizer,lr,None,'custom_loss2_lambda = '+str(lambda_))


def get_item(class_rgb_values,image_paths,mask_paths,preprocessing_fn):
        image = cv2.cvtColor(cv2.imread(image_paths), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_paths), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256),interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (256, 256),interpolation = cv2.INTER_NEAREST)
        # One-hot-encode the mask
        mask = pp.one_hot_encode(mask, class_rgb_values).astype('float')  
        # Apply augmentations
        
        # sample = ds.get_training_augmentation(image = image, mask = mask)
        # image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        
        preprocessing = ds.get_preprocessing(preprocessing_fn)
        sample = preprocessing(image = image, mask = mask)
        image, mask = sample['image'], sample['mask']
            
        return image, mask


def save_prediction(model_file1,model_file2,x_train_dir, y_train_dir,preprocessing_fn1,preprocessing_fn2,select_class_rgb_values,name):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_file1):
        model1 = torch.load(model_file1, map_location = DEVICE)
        model2 = torch.load(model_file2, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file can not be found !")
        return 
    predictions = []
    image_paths = [os.path.join(x_train_dir, image_id) for image_id in sorted(os.listdir(x_train_dir))]
    mask_paths = [os.path.join(y_train_dir, image_id) for image_id in sorted(os.listdir(y_train_dir))]
    for image_path,mask_path in zip(image_paths,mask_paths):
        image1,mask1 = get_item(select_class_rgb_values,image_path,mask_path,preprocessing_fn1)
        image2,mask2 = get_item(select_class_rgb_values,image_path,mask_path,preprocessing_fn2)
        if(not np.array_equal(mask1,mask2)):
          continue
        # print("Hello")
        x_tensor1 = torch.from_numpy(image1).to(DEVICE).unsqueeze(0)
        x_tensor2 = torch.from_numpy(image2).to(DEVICE).unsqueeze(0)
        pred_mask1 = model1(x_tensor1)
        pred_mask2 = model2(x_tensor2)
        pred_mask1 = pred_mask1.detach().squeeze().cpu().numpy()
        pred_mask1 = np.transpose(pred_mask1, (1, 2, 0))
        pred_mask2 = pred_mask2.detach().squeeze().cpu().numpy()
        pred_mask2 = np.transpose(pred_mask2, (1, 2, 0))
        gt_mask = np.transpose(mask1, (1, 2, 0))
        predictions.append([pred_mask1,pred_mask2,gt_mask])
    np.save(name+"_predictions_targets_train.npy",predictions)

def save_prediction(model_file1,model_file2,x_train_dir, y_train_dir,preprocessing_fn1,preprocessing_fn2,select_class_rgb_values,name):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_file1):
        model1 = torch.load(model_file1, map_location = DEVICE)
        model2 = torch.load(model_file2, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file can not be found !")
        return 
    predictions = []
    image_paths = [os.path.join(x_train_dir, image_id) for image_id in sorted(os.listdir(x_train_dir))]
    mask_paths = [os.path.join(y_train_dir, image_id) for image_id in sorted(os.listdir(y_train_dir))]
    for image_path,mask_path in zip(image_paths,mask_paths):
        image1,mask1 = get_item(select_class_rgb_values,image_path,mask_path,preprocessing_fn1)
        image2,mask2 = get_item(select_class_rgb_values,image_path,mask_path,preprocessing_fn2)
        if(not np.array_equal(mask1,mask2)):
          continue
        # print("Hello")
        x_tensor1 = torch.from_numpy(image1).to(DEVICE).unsqueeze(0)
        x_tensor2 = torch.from_numpy(image2).to(DEVICE).unsqueeze(0)
        pred_mask1 = model1(x_tensor1)
        pred_mask2 = model2(x_tensor2)
        pred_mask1 = pred_mask1.detach().squeeze().cpu().numpy()
        pred_mask1 = np.transpose(pred_mask1, (1, 2, 0))
        pred_mask2 = pred_mask2.detach().squeeze().cpu().numpy()
        pred_mask2 = np.transpose(pred_mask2, (1, 2, 0))
        gt_mask = np.transpose(mask1, (1, 2, 0))
        predictions.append([pred_mask1,pred_mask2,gt_mask])
    np.save(name+"_predictions_targets_train.npy",predictions)
