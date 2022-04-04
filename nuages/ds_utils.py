from os import listdir
import cv2
import numpy as np
import random
from matplotlib import pyplot

################################################################################
# Loading and stocking paths of datasets
#
################################################################################
def load_paths(path):
    pathA = path + 'sans_nuage/'
    pathB = path + 'avec_nuage/'
    datasetA, datasetB = list(), list()
    for file in listdir(pathA):
        datasetA.append(pathA+file)
    for file in listdir(pathB):
        datasetB.append(pathB+file)
    return datasetA, datasetB


################################################################################
# Data preparation for CNN
#
################################################################################
def get_training_data(dataset, to_do, n_samples, augmentation=True):
    # Unpack dataset and to_do list
    resized_shape = to_do['img_size']
    spatial_aug_ratio = to_do['spatial_aug_ratio']
    info_aug_ratio = to_do['info_aug_ratio']

    # Declare empty list
    X = list()

    # Load images and identifications
    ix = np.random.randint(0, len(dataset), n_samples)
    for i in ix:
        # Prepare image and prediction
        img = cv2.imread(dataset[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img / 255.0

        # Resize if needed and append to batch list
        if not resized_shape == img.shape[0:2]:
            X.append( resize_data(img, resized_shape) )
        else:
            X.append(img)

    if augmentation:
        # Do flips and rotations
        for i in range(len(X)):
            if np.random.uniform(0,1) < spatial_aug_ratio:
                modif = np.random.randint(6)
                if modif == 0:
                    X[i] = rotate_data(X[i], cv2.ROTATE_90_CLOCKWISE)
                elif modif == 1:
                    X[i] = rotate_data(X[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif modif == 2:
                    X[i] = rotate_data(X[i], cv2.ROTATE_180)
                elif modif == 3:
                    X[i] = flip_data(X[i], 0)
                elif modif == 4:
                    X[i] = flip_data(X[i], 1)

        # Do others augmentation functions
        for i in range(len(X)):
            if np.random.uniform(0,1) < info_aug_ratio:
                modif = np.random.randint(0)
                if modif == 0:
                    X[i] = apply_random_bright(X[i])
                elif modif == 4:
                    X[i] = shuffle_colors(X[i])

    #X2[0] = X2[0].reshape((65536, 1))
    x_add = 0.5
    x_factor = 0.5

    # Transform as arrays for tensorflow computations
    X = (np.asarray(X) - x_add) / x_factor
    return X


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


################################################################################
# Summaryzing and saving functions
#
################################################################################
def summarize_performance(step, g_model, dataset, aug_params, save_path, epoch, n_samples=3):
    save_name = 'tanh'
    img = get_training_data(dataset, aug_params, n_samples)
    pred, _, pred_attn = g_model.predict(img)

    for i in range(n_samples):
        # Prepare images
        x_add = 1.0
        x_factor = 127.5
        img_to_print = cv2.cvtColor(np.squeeze(((img[i] + x_add) * x_factor)).astype(np.uint8), cv2.COLOR_LAB2RGB)
        pred_to_print = cv2.cvtColor(np.squeeze(((pred[i] + x_add) * x_factor)).astype(np.uint8), cv2.COLOR_LAB2RGB)
        #pred_gen = cv2.cvtColor(np.squeeze(((pred_gen + to_add) * to_factor)).astype(np.uint8), cv2.COLOR_LAB2RGB)
    
        # Plot initial image
        pyplot.subplot(3, n_samples, i+1)
        pyplot.axis('off')
        pyplot.imshow(img_to_print)
        # Plot disease segmentation (attention network)
        pyplot.subplot(3, n_samples, 1+n_samples+i)
        pyplot.axis('off')
        pyplot.imshow(np.squeeze(pred_attn[i]))
        # Plot transformation
        pyplot.subplot(3, n_samples, 1+2*n_samples+i)
        pyplot.axis('off')
        pyplot.imshow(pred_to_print)
    pyplot.savefig(save_path + 'images/' + save_name + '_' + str(epoch) + '.png')


################################################################################
# Data Augmentation
#
################################################################################
def apply_random_bright(img):
    random_bright = .5+np.random.uniform()
    img = np.array(img, dtype=np.float64)
    img[:,:,2] = img[:,:,2] * random_bright
    img[:,:,2][img[:,:,2]>255] = 255
    img = np.array(img, dtype = np.uint8)
    return img

def shuffle_colors(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    new_colors = np.arange(3)
    while (new_colors == [0,1,2]).all():
        np.random.shuffle(new_colors)
    img = img[:,:,new_colors]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

def resize_data(img, resized_shape, interpolation=cv2.INTER_AREA):
    one_by_one = False
    if len(img.shape) == 3:
        if img.shape[2] > 4:
            one_by_one = True

    if one_by_one:
        resized_img = np.zeros((resized_shape[0], resized_shape[1], img.shape[2]))
        for i in range(img.shape[2]):
            resized_img[:,:,i] = cv2.resize(img[:,:,i], resized_shape, interpolation=interpolation)
    else :
        resized_img = cv2.resize(img, resized_shape, interpolation=interpolation)
    return resized_img

def rotate_data(img, angle):
    one_by_one = False
    if len(img.shape) == 3:
        if img.shape[2] > 4:
            one_by_one = True

    if one_by_one:
        rotated_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        for i in range(img.shape[2]):
            rotated_img[:,:,i] = cv2.rotate(img[:,:,i], angle)
    else:
        rotated_img = cv2.rotate(img, angle)
    return rotated_img

def flip_data(img, direction):
    one_by_one = False
    if len(img.shape) == 3:
        if img.shape[2] > 4:
            one_by_one = True

    if one_by_one:
        fliped_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        for i in range(img.shape[2]):
            fliped_img[:,:,i] = cv2.flip(img[:,:,i], direction)
    else:
        fliped_img = cv2.flip(img, direction)
    return fliped_img

def crop_to_min_true(img, pred, size=(256,256)):
    # Find true values
    idx_1 = np.where( pred  > 0 )

    # Crop to min and max true values
    img = img[np.min(idx_1[0]):np.max(idx_1[0]), np.min(idx_1[1]):np.max(idx_1[1]), :]
    pred = pred[np.min(idx_1[0]):np.max(idx_1[0]), np.min(idx_1[1]):np.max(idx_1[1])]

    # Resize to size
    img_r = resize_data(img, size)
    pred_r = resize_data(pred, size, interpolation=cv2.INTER_NEAREST)
    return img_r, pred_r

def create_roll_distortion(img, pred):
    # Initialize distorted image and prediction
    dst_img = np.copy(img)
    dst_pred = np.copy(pred)

    # Disrtortion parameters
    transfo = np.random.randint(3)
    max_offset = int(((0.5 - 0.25) * np.random.random() + 0.25) * dst_img.shape[1])
    curve_end = (0.75 - 0.25) * np.random.random() + 0.25
    deg = (3 - 1.5) * np.random.random() + 1.5
    v_dir = np.random.randint(2)
    h_dir = np.random.randint(2)

    # Create rolling vectors
    if transfo == 0:
        plant_offset = 2*max_offset/3
        if h_dir == 0:
            offset_values = np.concatenate([np.power(np.linspace(0,1,int(dst_img.shape[1]*curve_end)), deg)*max_offset,
                max_offset*np.ones(1+int(dst_img.shape[1]-dst_img.shape[1]*curve_end))]) - plant_offset
        elif h_dir == 1:
            offset_values = np.concatenate([max_offset*np.ones(1+int(dst_img.shape[1]-dst_img.shape[1]*curve_end)),
                np.power(np.abs(np.linspace(-1,0,int(dst_img.shape[1]*curve_end))), deg)*max_offset]) - plant_offset
    elif transfo == 1:
        plant_offset = max_offset/5
        if h_dir == 0:
            offset_values = (np.cos(np.linspace(-np.pi/2, np.pi/2, dst_img.shape[1])) * max_offset) - 4*plant_offset
        elif h_dir == 1:
            offset_values = (np.clip(1-np.cos(np.linspace(-np.pi/2, np.pi/2, dst_img.shape[1])), 0, 1) * max_offset) - plant_offset
    elif transfo == 2:
        offset_mod = max_offset / 5
        offset_values = (np.cos(np.linspace(-np.pi*deg, np.pi*deg, dst_img.shape[1])) * offset_mod)

    # Distort original image and prediction
    if len(dst_img.shape) == 2:
        img_zdim = 1
    else:
        img_zdim = img.shape[2]

    if len(dst_pred.shape) == 2:
        pred_zdim = 1
    else:
        pred_zdim = img.shape[2]

    for i in range(dst_img.shape[1]):
        if v_dir == 0:
            if offset_values[i] > 0:
                dst_img[:,i] = np.concatenate([np.zeros((int(offset_values[i]), img_zdim)), dst_img[0:(dst_img.shape[0]-int(offset_values[i])),i]])
                dst_pred[:,i] = np.concatenate([np.zeros((int(offset_values[i]))), dst_pred[0:(dst_pred.shape[0]-int(offset_values[i])),i]])
            else:
                dst_img[:,i] = np.concatenate([dst_img[int(-offset_values[i]):,i], np.zeros((int(-offset_values[i]), img_zdim))])
                dst_pred[:,i] = np.concatenate([dst_pred[int(-offset_values[i]):,i], np.zeros((int(-offset_values[i])))])
        elif v_dir == 1:
            if offset_values[i] > 0:
                dst_img[i,:] = np.concatenate([np.zeros((int(offset_values[i]), img_zdim)), dst_img[i,0:(dst_img.shape[0]-int(offset_values[i]))]])
                dst_pred[i,:] = np.concatenate([np.zeros((int(offset_values[i]))), dst_pred[i,0:(dst_pred.shape[0]-int(offset_values[i]))]])
            else:
                dst_img[i,:] = np.concatenate([dst_img[i,int(-offset_values[i]):], np.zeros((int(-offset_values[i]), img_zdim))])
                dst_pred[i,:] = np.concatenate([dst_pred[i,int(-offset_values[i]):], np.zeros((int(-offset_values[i])))])

    # Crop to minimum ground truth true value
    dst_img, dst_pred = crop_to_min_true(dst_img, dst_pred)

    return dst_img, dst_pred


def create_focal_distortion(img, pred):
    # Initialize distorted image and prediction
    dst_img = np.copy(img)
    dst_pred = np.copy(pred)

    # Disrtortion parameters
    center_x = (7*dst_img.shape[0]/8 - dst_img.shape[0]/8) * np.random.random() + dst_img.shape[0]/8
    center_y = (7*dst_img.shape[1]/8 - dst_img.shape[0]/8) * np.random.random() + dst_img.shape[1]/8
    focal_length_1 = (250 - 190) * np.random.random() + 190
    focal_length_2 = (250 - 190) * np.random.random() + 190
    mtx = np.array([[focal_length_1, 0.0, center_x],
                    [0.0, focal_length_2, center_y],
                    [0.0, 0.0, 1.0]])
    unaligned_coef = [0.1, 0.1]
    radial_coef = [1.0, 1.0, 1.0]
    dst = (radial_coef[0], radial_coef[1], unaligned_coef[0], unaligned_coef[1], radial_coef[2])

    # Distort original image and prediction
    dst_img = cv2.undistort(dst_img, mtx, dst)
    dst_pred = cv2.undistort(dst_pred, mtx, dst)

    # Crop to minimum ground truth true value
    dst_img, dst_pred = crop_to_min_true(dst_img, dst_pred)

    return dst_img, dst_pred


def add_leaves_to_plant(img, pred, dataset, segmentation_type):
    success = False
    cpt_success = 0
    while not success:
        try:
            # Prepare lists
            centroids = []
            imgs = []
            preds = []
            n_plants = np.random.randint(2,4)
            N_leaves = np.zeros(n_plants, dtype=np.uint8)
            new_shape = [0, 0]
            for i in range(n_plants):
                # Prepare image and prediction
                rand_leave = np.random.randint(len(dataset[0]))
                img_to_store = np.load(dataset[0][rand_leave][:-7] + 'rgb.npy')
                pred_to_store = np.load(dataset[1][rand_leave])
                img_to_store = resize_data(img_to_store, img.shape[:-1])
                pred_to_store = resize_data(pred_to_store, (img.shape[0], img.shape[1]))

                # Store images and predictions
                imgs.append(img_to_store)
                preds.append(pred_to_store)

                # Calculate centroids
                M = cv2.moments(np.sum(pred_to_store, axis=-1))
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append([cX, cY])

                # Calculate number of leaves
                for j in range(32):
                    if np.sum(pred_to_store[:,:,j]) > 0:
                        N_leaves[i] += 1

            # Choose a random centroid for new plant
            pos = np.random.randint(len(N_leaves))
            diffs = []
            for i in range(len(N_leaves)):
                if i == pos:
                    diffs.append([0, 0])
                else:
                    diffs.append([centroids[i][0] - centroids[pos][0], centroids[i][1] - centroids[pos][1]])

            # Create new image and prediction
            rr_freq = 2
            new_img = np.zeros(imgs[pos].shape)
            new_pred = np.zeros(preds[pos].shape)
            plant_idx = None
            new_N_leaves = np.max(N_leaves)
            for i in reversed(range(new_N_leaves)):
                # Reroll a new plant
                if i%rr_freq == 0 or plant_idx == None:
                    plant_idx = np.random.randint(len(N_leaves))
                    # Make sure plant has enough leaves
                    while i > N_leaves[plant_idx]:
                        plant_idx = np.random.randint(len(N_leaves))

                # Keep only the portion of a leave that fit on the image size
                idx_1 = np.where( preds[plant_idx][:,:,i]  > 0 )
                idx_delete = []
                for j in reversed(range(len(idx_1[0]))):
                    if not ((new_img.shape[0] > idx_1[0][j] + diffs[plant_idx][1]) and (idx_1[0][j] + diffs[plant_idx][1] >= 0) and (new_img.shape[1] > idx_1[1][j]- diffs[plant_idx][0]) and (idx_1[1][j] - diffs[plant_idx][0] >= 0)):
                        idx_delete.append(j)
                idx_copy = [[],[]]
                idx_copy[0] = np.delete(idx_1[0], idx_delete)
                idx_copy[1] = np.delete(idx_1[1], idx_delete)

                # Print leave on new image and add prediction to tensor
                new_img[idx_copy[0] + diffs[plant_idx][1], idx_copy[1] - diffs[plant_idx][0], :] = imgs[plant_idx][idx_copy[0], idx_copy[1], :]
                new_pred[idx_copy[0] + diffs[plant_idx][1], idx_copy[1] - diffs[plant_idx][0], i] = preds[plant_idx][:,:,i][idx_copy[0], idx_copy[1]]

            # Add ground to new img
            completed_pred = np.zeros(new_img.shape[0:-1])
            idx_pred = np.where( np.sum(new_pred, axis=-1) > 0 )
            completed_pred[idx_pred[0], idx_pred[1]] = 1
            idx = 0
            while (np.sum(completed_pred-1) < 0) and (idx < len(N_leaves)):
                # Replace empty spaces by ground of images
                idx_ground = np.where( np.sum(np.dstack([completed_pred, preds[idx]]), axis=-1) == 0 )
                if len(idx_ground) == 0:
                    break
                new_img[idx_ground[0], idx_ground[1], :] = imgs[idx][idx_ground[0], idx_ground[1], :]
                completed_pred[idx_ground[0], idx_ground[1]] = 1
                idx += 1

            # Find centroid of plant
            idx_1 = np.where( np.sum(new_pred, axis=-1)  > 0 )
            new_seg = np.zeros([new_pred.shape[0], new_pred.shape[1]])
            new_seg[idx_1[0], idx_1[1]] = 1
            M = cv2.moments(new_seg)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            img_center = [int(cY), int(cX)]

            # Calculate centroids
            dist_to_center = []
            angles_order = []
            for i in range(new_N_leaves):
                # Find centroid of leaf and calculate distance from image center
                M = cv2.moments(new_pred[:, :, i])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist_to_center.append( np.sqrt( np.square(cY-img_center[0]) + np.square(cX-img_center[1]) ) )
                #angles_order.append( np.arctan2(cY-img_center[0], cX-img_center[1]) )

            # Sort prediction tensor (leaves at center have smaller depth)
            leaves_sorting = np.argsort(dist_to_center)
            new_pred[:,:,:len(leaves_sorting)] = new_pred[:,:,leaves_sorting]

            # Do watershed transform if needed
            if segmentation_type == 'wleaf':
                prediction = np.zeros((new_pred.shape[0], new_pred.shape[1]))
                n_max = new_pred.shape[2]
                for i in range(n_max):
                    pred_to_add = np.where( new_pred[:,:,i] > 0 )
                    if len(pred_to_add) == 2:
                        prediction[pred_to_add[0], pred_to_add[1]] = i+1
                new_pred = prediction

            # Prepare data for cnn
            idx_1 = np.where( new_pred > 0 )
            new_seg = np.zeros(new_pred.shape)
            new_seg[idx_1[0], idx_1[1]] = 1
            new_img = np.dstack([new_img, new_seg])
            new_pred = np.abs(33-new_pred)*new_seg

            # Crop to minimum ground truth true value
            new_img, new_pred = crop_to_min_true(new_img, new_pred)

            # Success flag
            success = True
        except:
            cpt_success += 1
            if cpt_success < 15:
                success = False
            else:
                print('Can\'t do shit with add_leaves_to_plant function... returning with errors')
                success = True

    return new_img, new_pred
