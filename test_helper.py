import torch
import segmentation_models_pytorch as smp
import csv
import dataset as ds
import preprocess as pp
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import cv2

def crop_image(image, target_image_dims = [1500, 1500, 3]):
   
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]


def test(x_test_dir,y_test_dir,preprocessing_fn,select_class_rgb_values,prediction_dir,model_file,select_classes,loss,metrics):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_file):
        model = torch.load(model_file, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file can not be found !")
        return 
    test_dataset = ds.BuildingsDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation = ds.get_validation_augmentation(), 
        preprocessing = ds.get_preprocessing(preprocessing_fn),
        class_rgb_values = select_class_rgb_values,
    )
    test_dataloader = DataLoader(test_dataset)
    # test_dataset_vis = ds.BuildingsDataset(
    #                                     x_test_dir, y_test_dir, 
    #                                     augmentation = ds.get_validation_augmentation(),
    #                                     class_rgb_values = select_class_rgb_values,
    #                                     )
    # Get a random test image / mask index
    # random_idx = random.randint(0, len(test_dataset_vis) - 1)
    # image, mask = test_dataset_vis[random_idx]

    # pp.visualize(
    #     original_image = image,
    #     ground_truth_mask = pp.colour_code_segmentation(pp.reverse_one_hot(mask), select_class_rgb_values),
    #     one_hot_encoded_mask = pp.reverse_one_hot(mask)
    # )
    # sample_preds_folder = prediction_dir+"/"
    # if not os.path.exists(sample_preds_folder):
    #     os.makedirs(sample_preds_folder)
    total_accuracy = 0
    for idx in range(len(test_dataset)):

        image, gt_mask = test_dataset[idx]
        # image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to building
        # pred_building_heatmap = pred_mask[:,:,select_classes.index('building')]
        # pred_mask = crop_image(pp.colour_code_segmentation(pp.reverse_one_hot(pred_mask), select_class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        print(pred_mask.shape,gt_mask.shape)
        
        pred_mask_onehot = pred_mask
        for i in range(len(pred_mask)):
          for j in range(len(pred_mask[i])):
            pred_mask_onehot[i][j] = np.where(pred_mask[i][j]>0.5, 1, 0)
        accuracy = 0
        for i in range(len(pred_mask)):
          for j in range(len(pred_mask[i])):
            if(gt_mask[i][j][0] == 0 and gt_mask[i][j][1] == 0):
              accuracy += 1
            elif(np.array_equal(gt_mask[i][j],pred_mask_onehot[i][j])):
              accuracy += 1
        total_accuracy += (accuracy/(len(pred_mask)*(len(pred_mask[0]))))

    accuracy = total_accuracy/(len(test_dataset))
        # gt_mask = crop_image(pp.colour_code_segmentation(pp.reverse_one_hot(gt_mask), select_class_rgb_values))
        # cv2.imwrite(os.path.join(sample_preds_folder, sample_preds_folder+"_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
        
        # pp.visualize(
        #     original_image = image_vis,
        #     ground_truth_mask = gt_mask,
        #     predicted_mask = pred_mask,
        #     predicted_building_heatmap = pred_building_heatmap
        # )
        # intersection = np.logical_and(gt_mask, pred_mask)
        # union = np.logical_or(gt_mask, pred_mask)
        # iou_score = np.sum(intersection) / np.sum(union)
        # iouscores += iou_score

    # print("Mean IoU Score: "+str(iouscores/len(test_dataset)))
    # f = open('./RESULT/TEST/loss_iou.txt','a')

    # f.write(model_file+" "+str(iouscores/len(test_dataset)))
    # f.write("\n")
    # f.close()


    test_epoch = smp.utils.train.ValidEpoch(
                                            model,
                                            loss = loss, 
                                            metrics = metrics, 
                                            # device = DEVICE,
                                            verbose = True,
                                            )
    valid_logs = test_epoch.run(test_dataloader)
    print("Accuracy on test dataset : \n",accuracy)

def combine_predictions(prediction_file_1,prediction_file_2,gt_file_1,gt_file_2,em1,em2):
    predictions_model1 = np.load(prediction_file_1)
    predictions_model2 = np.load(prediction_file_2)
    gt_model1 = np.load(gt_file_1)
    gt_model2 = np.load(gt_file_2)
    total_iou_score = 0
    total_accuracy_score = 0
    for i in range(len(predictions_model1)):
        pred1 = predictions_model1[i]
        pred2 = predictions_model2[i]
        gt_mask = gt_model1[i]
        final_mask = pred1
        gamma = em1+em2
        for i in range(len(pred1)):
            for j in range(len(pred1[i])):
                x = pred1[i][j]*(em1/gamma)
                y = pred2[i][j]*(em2/gamma)
                final_mask[i][j] = x+y
        pred_mask_onehot = final_mask
        for i in range(len(final_mask)):
          for j in range(len(final_mask[i])):
            pred_mask_onehot[i][j] = np.where(final_mask[i][j]>0.5, 1, 0)

        intersection = np.logical_and(pred_mask_onehot, gt_mask)
        union = np.logical_or(pred_mask_onehot, gt_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        total_iou_score += iou_score
        equality = pred_mask_onehot == gt_mask
        n = final_mask.shape[0]*final_mask.shape[1]*final_mask.shape[2]
        total_accuracy_score += equality.sum()/n
    print("IOU SCORE ",total_iou_score/len(predictions_model1))
    print("ACCURACY SCORE ",total_accuracy_score/len(predictions_model1))


def save_prediction(name,model_file,x_test_dir,y_test_dir,preprocessing_fn,select_class_rgb_values,loss,metric):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_file):
        model = torch.load(model_file, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file can not be found !")
        return 
    test_dataset = ds.BuildingsDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation = ds.get_validation_augmentation(), 
        preprocessing = ds.get_preprocessing(preprocessing_fn),
        class_rgb_values = select_class_rgb_values,
    )
    predictions = []
    targets = []
    for idx in range(len(test_dataset)):

        image, gt_mask = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        pred_mask_onehot = pred_mask
        predictions.append(pred_mask_onehot)
        targets.append(gt_mask)
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    np.save(name+"_predictions_"+".npy",predictions)
    np.save(name+"_targets_"+".npy",targets)
  



def ensemble_result(model_file1,model_file2,x_test_dir,y_test_dir,preprocessing_fn,select_class_rgb_values,loss,metric):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_file1):
        model1 = torch.load(model_file1, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file 1 can not be found !")
    if os.path.exists(model_file2):
        model2 = torch.load(model_file1, map_location = DEVICE)
        print('Loaded UNet model from this run.')
    else:
        print("Model file 2 can not be found !")
    test_dataset = ds.BuildingsDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation = ds.get_validation_augmentation(), 
        preprocessing = ds.get_preprocessing(preprocessing_fn),
        class_rgb_values = select_class_rgb_values,
    )
    total_accuracy = 0
    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask1 = model1(x_tensor)
        pred_mask2 = model2(x_tensor)
        pred_mask1 = pred_mask1.detach().squeeze().cpu().numpy()
        pred_mask2 = pred_mask2.detach().squeeze().cpu().numpy()
        pred_mask1 = np.transpose(pred_mask1, (1, 2, 0))
        pred_mask2 = np.transpose(pred_mask2, (1, 2, 0))
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        final_mask = pred_mask1
        gamma = em1+em2
        for i in range(len(pred_mask1)):
            for j in range(len(pred_mask1[i])):
                x = pred_mask1*(em1/gamma)
                y = pred_mask2*(em2/gamma)
                final_mask = x+y
        print(final_mask)
        pred_mask_onehot = final_mask
        for i in range(len(final_mask)):
          for j in range(len(final_mask[i])):
            pred_mask_onehot[i][j] = np.where(final_mask[i][j]>0.5, 1, 0)
        accuracy = 0
        for i in range(len(final_mask)):
          for j in range(len(final_mask[i])):
            if(gt_mask[i][j][0] == 0 and gt_mask[i][j][1] == 0):
              accuracy += 1
            elif(np.array_equal(gt_mask[i][j],pred_mask_onehot[i][j])):
              accuracy += 1
        total_accuracy += (accuracy/(len(final_mask)*(len(final_mask[0]))))


def ensemble_result_(prediction_target_file,model):
    prediction_target = np.load(prediction_target_file)
    for i in range(len(prediction_target)):
        print(prediction_target[i][0].shape)
        print(prediction_target[i][1].shape)
        print(prediction_target[i][2].shape)
        print("---------------->")

