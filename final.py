import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_iou(y_true, y_pred, num_classes):
    iou = []
    for cls in range(num_classes):
        true_class = y_true == cls
        pred_class = y_pred == cls
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        if union == 0:
            iou.append(float('nan'))  # If no union, set IoU to NaN
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)  # Calculate mean while ignoring NaN values

def calculate_pix_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return float('nan')  # If no union, set IoU to NaN
    else:
        return intersection / union


def calculate_map(y_true, y_pred, num_classes):
    aps = []
    for cls in range(num_classes):
        true_class = (y_true == cls).astype(int).ravel()
        pred_class = (y_pred == cls).astype(int).ravel()
        if true_class.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(true_class, pred_class)
        aps.append(average_precision_score(true_class, pred_class))
    return np.nanmean(aps)  # Calculate mean while ignoring NaN values


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0
    return image

def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask = tf.image.resize(mask, [256, 256])
    mask = mask / 255.0
    return mask

def get_image_files(dataset_path):
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file_name))
    return image_files




def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.equal(y_true, y_pred)
    accuracy = np.mean(correct_predictions)
    return accuracy



import tensorflow as tf

class Network(tf.keras.Model):
    def __init__(self, batch_size=32):
        super(Network, self).__init__()
        self.batch_size = batch_size

        # Encoder layers
        self.conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='SAME')

        self.conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='SAME')

        self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), padding='SAME')

        self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D((2, 2), padding='SAME')

        self.conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D((2, 2), padding='SAME')

        # Conditional layers
        self.Bconv1_1 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='SAME', activation='relu')
        self.Bconv1_2 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='SAME', activation='relu')
        self.Bpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')

        self.Bconv2_1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.Bconv2_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.Bpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')

        self.Bconv3_1 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu')
        self.Bpool3 = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')

        self.Bconv4_1 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', activation='relu')
        self.Bpool4 = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')

        self.Bconv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', activation='relu')
        self.Bpool5 = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')

        self.Bconv6 = tf.keras.layers.Conv2D(512, kernel_size=2, activation='relu')

        # Decoder layers
        self.Dconv1_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', activation='relu')
        self.Dconv1_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', activation='relu')
        self.upsam1 = tf.keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='SAME', activation='relu')

        self.Dconv2_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', activation='relu')
        self.Dconv2_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', activation='relu')
        self.upsam2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='SAME', activation='relu')

        self.Dconv3_1 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', activation='relu')
        self.Dconv3_2 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', activation='relu')
        self.upsam3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='SAME', activation='relu')

        self.Dconv4_1 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu')
        self.Dconv4_2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu')
        self.upsam4 = tf.keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='SAME', activation='relu')

        self.Dconv5_1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.Dconv5_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.upsam5 = tf.keras.layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='SAME', activation='relu')

        # Final output layer
        
       
        # Additional convolutional layers before the final output
               # Additional convolutional layers before the final output
        self.additional_conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.additional_conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')
        self.additional_conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu')

        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Dropout layer for regularization
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Final output layer with sigmoid activation for binary segmentation
        self.output_conv = tf.keras.layers.Conv2D(1, kernel_size=1, padding='SAME', activation='sigmoid')


    def call(self, Input1, Input2):
        # Encoder
        x1 = self.conv1_1(Input1)
        x1 = self.conv1_2(x1)
        pool1_shape = tf.shape(x1)
        x1 = self.pool1(x1)

        x1 = self.conv2_1(x1)
        x1 = self.conv2_2(x1)
        pool2_shape = tf.shape(x1)
        x1 = self.pool2(x1)

        x1 = self.conv3_1(x1)
        x1 = self.conv3_2(x1)
        pool3_shape = tf.shape(x1)
        x1 = self.pool3(x1)

        x1 = self.conv4_1(x1)
        x1 = self.conv4_2(x1)
        pool4_shape = tf.shape(x1)
        x1 = self.pool4(x1)

        x1 = self.conv5_1(x1)
        x1 = self.conv5_2(x1)
        pool5_shape = tf.shape(x1)
        x1 = self.pool5(x1)

        # Conditional layers
        x2 = self.Bconv1_1(Input2)
        x2 = self.Bconv1_2(x2)
        x2 = self.Bpool1(x2)

        x2 = self.Bconv2_1(x2)
        x2 = self.Bconv2_2(x2)
        x2 = self.Bpool2(x2)

        x2 = self.Bconv3_1(x2)
        x2 = self.Bpool3(x2)

        x2 = self.Bconv4_1(x2)
        x2 = self.Bpool4(x2)

        x2 = self.Bconv5_1(x2)
        x2 = self.Bpool5(x2)

        x2 = self.Bconv6(x2)

        # Decoder
        efused_1 = tf.concat([x1, tf.image.resize(x2, tf.shape(x1)[1:3])], axis=-1)
        x1 = self.Dconv1_1(efused_1)
        x1 = self.Dconv1_2(x1)
        x1 = self.upsam1(x1)

        efused_2 = tf.concat([x1, tf.image.resize(x2, tf.shape(x1)[1:3])], axis=-1)
        x1 = self.Dconv2_1(efused_2)
        x1 = self.Dconv2_2(x1)
        x1 = self.upsam2(x1)

        efused_3 = tf.concat([x1, tf.image.resize(x2, tf.shape(x1)[1:3])], axis=-1)
        x1 = self.Dconv3_1(efused_3)
        x1 = self.Dconv3_2(x1)
        x1 = self.upsam3(x1)

        efused_4 = tf.concat([x1, tf.image.resize(x2, tf.shape(x1)[1:3])], axis=-1)
        x1 = self.Dconv4_1(efused_4)
        x1 = self.Dconv4_2(x1)
        x1 = self.upsam4(x1)

        efused_5 = tf.concat([x1, tf.image.resize(x2, tf.shape(x1)[1:3])], axis=-1)
        x1 = self.Dconv5_1(efused_5)
        x1 = self.Dconv5_2(x1)
        x1 = self.upsam5(x1)

        # Final output
        x1 = self.additional_conv1(x1)
        x1 = self.additional_conv2(x1)
        x1 = self.additional_conv3(x1)

        x1 = self.batch_norm(x1)
        x1 = self.dropout(x1)

        output = self.output_conv(x1)

        return output
        return x1

    
# Define the learning rate scheduler
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Example of categorical cross-entropy loss function
def compute_loss(predictions, masks):
    loss = tf.keras.losses.BinaryCrossentropy()(masks, predictions)
    return loss

# Example of data augmentation function
def augment(image, mask):
    # Example augmentation: horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

import os
from sklearn.model_selection import train_test_split

def main():
    target_dataset = "/kaggle/input/final-dataset-thesis/Datasets/classes/target"
    query_dataset = "/kaggle/input/final-dataset-thesis/Datasets/classes/query"
    mask_dataset = "/kaggle/input/final-dataset-thesis/Datasets/classes/masks"
    
    target_folders = [os.path.join(target_dataset, folder) for folder in os.listdir(target_dataset) if os.path.isdir(os.path.join(target_dataset, folder))]
    query_folders = [os.path.join(query_dataset, folder) for folder in os.listdir(query_dataset) if os.path.isdir(os.path.join(query_dataset, folder))]
    mask_folders = [os.path.join(mask_dataset, folder) for folder in os.listdir(mask_dataset) if os.path.isdir(os.path.join(mask_dataset, folder))]
    
    folder_mapping_query = {os.path.basename(folder): folder for folder in query_folders}
    folder_mapping_mask = {os.path.basename(folder): folder for folder in mask_folders}

    # Lists to store metrics across epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    mIoU_list = []
    mPixIoU_list = []
    mAP_list = []
    num_classes= 29

    for epoch in range(1):  # Number of epochs
        for target_folder in target_folders:
            target_folder_name = os.path.basename(target_folder)
            query_folder = folder_mapping_query.get(target_folder_name)
            mask_folder = folder_mapping_mask.get(target_folder_name)

            if not query_folder or not mask_folder:
                print(f"No matching query or mask folder found for target folder: {target_folder_name}")
                continue
            
            # Print selected folders
            print(f"Epoch {epoch}, Selected folders - Target: {target_folder}, Query: {query_folder}, Mask: {mask_folder}")

            target_files = get_image_files(target_folder)
            query_files = get_image_files(query_folder)
            mask_files = get_image_files(mask_folder)
            
            target_files.sort()
            query_files.sort()
            mask_files.sort()
            
            if not target_files or not query_files or not mask_files:
                continue

            target_images = [load_image(file) for file in target_files]
            masks = [load_mask(file) for file in mask_files]

            target_images = np.array(target_images)
            masks = np.array(masks)

            query_image = load_image(query_files[0])
            query_image = tf.expand_dims(query_image, axis=0)

            train_indices, temp_indices = train_test_split(range(len(target_images)), test_size=0.3, random_state=42)
            val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

            train_images = tf.gather(target_images, train_indices)
            train_masks = tf.gather(masks, train_indices)
            val_images = tf.gather(target_images, val_indices)
            val_masks = tf.gather(masks, val_indices)
            test_images = tf.gather(target_images, test_indices)
            test_masks = tf.gather(masks, test_indices)

            # Initialize model and optimizer within the epoch loop
            model = Network()
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0
            total_mIoU = 0.0
            total_mPixIoU = 0.0
            total_mAP = 0.0
            print(f"Epoch {epoch}, Target Folder: {target_folder_name}")

            # Training
            for target, mask in zip(train_images, train_masks):
                target, mask = augment(target, mask)
                target = tf.expand_dims(target, axis=0)
                mask = tf.expand_dims(mask, axis=0)

                with tf.GradientTape() as tape:
                    predictions = model(target, query_image)
                    loss = compute_loss(predictions, mask)
                    accuracy = calculate_accuracy(mask.numpy(), predictions.numpy() > 0.5)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                total_loss += loss.numpy()
                total_accuracy += accuracy
                total_mIoU += calculate_iou(mask.numpy().squeeze(), (predictions.numpy().squeeze() > 0.5), num_classes)
                total_mPixIoU += calculate_pix_iou(mask.numpy().squeeze(), (predictions.numpy().squeeze() > 0.5))
                total_mAP += calculate_map(mask.numpy().squeeze(), (predictions.numpy().squeeze() > 0.5), num_classes)
                num_batches += 1

                print(f"Batch {num_batches}, Loss: {loss.numpy()}, Accuracy: {accuracy}")

            average_loss = total_loss / num_batches
            average_accuracy = total_accuracy / num_batches
            average_mIoU = total_mIoU / num_batches
            average_mPixIoU = total_mPixIoU / num_batches
            average_mAP = total_mAP / num_batches
            print(f"Epoch {epoch}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}, mIoU: {average_mIoU}, mPixIoU: {average_mPixIoU}, mAP: {average_mAP}")

            # Validation
            total_val_loss = 0.0
            total_val_accuracy = 0.0
            total_val_mIoU = 0.0
            total_val_mPixIoU = 0.0
            total_val_mAP = 0.0
            num_val_batches = 0

            for val_image, val_mask in zip(val_images, val_masks):
                val_image = tf.expand_dims(val_image, axis=0)
                val_predictions = model(val_image, query_image)
                val_loss = compute_loss(val_predictions, tf.expand_dims(val_mask, axis=0))
                val_accuracy = calculate_accuracy(val_mask.numpy(), val_predictions.numpy() > 0.5)

                total_val_loss += val_loss.numpy()
                total_val_accuracy += val_accuracy
                total_val_mIoU += calculate_iou(val_mask.numpy().squeeze(), (val_predictions.numpy().squeeze() > 0.5), num_classes)
                total_val_mPixIoU += calculate_pix_iou(val_mask.numpy().squeeze(), (val_predictions.numpy().squeeze() > 0.5))
                total_val_mAP += calculate_map(val_mask.numpy().squeeze(), (val_predictions.numpy().squeeze() > 0.5), num_classes)
                num_val_batches += 1

            average_val_loss = total_val_loss / num_val_batches
            average_val_accuracy = total_val_accuracy / num_val_batches
            average_val_mIoU = total_val_mIoU / num_val_batches
            average_val_mPixIoU = total_val_mPixIoU / num_val_batches
            average_val_mAP = total_val_mAP / num_val_batches
            print(f"Epoch {epoch}, Validation Average Loss: {average_val_loss}, Validation Average Accuracy: {average_val_accuracy}, Validation mIoU: {average_val_mIoU}, Validation mPixIoU: {average_val_mPixIoU}, Validation mAP: {average_val_mAP}")
            
            # Append metrics to lists
            train_losses.append(average_loss)
            train_accuracies.append(average_accuracy)
            val_losses.append(average_val_loss)
            val_accuracies.append(average_val_accuracy)
            mIoU_list.append(average_mIoU)
            mPixIoU_list.append(average_mPixIoU)
            mAP_list.append(average_mAP)
            
    # Testing Loop after all epochs and target folders
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    total_test_mIoU = 0.0
    total_test_mPixIoU = 0.0
    total_test_mAP = 0.0
    num_test_batches = 0

    for test_image, test_mask in zip(test_images, test_masks):
        test_image = tf.expand_dims(test_image, axis=0)
        test_predictions = model(test_image, query_image)
        test_loss = compute_loss(test_predictions, tf.expand_dims(test_mask, axis=0))
        test_accuracy = calculate_accuracy(test_mask.numpy(), test_predictions.numpy() > 0.5)

        total_test_loss += test_loss.numpy()
        total_test_accuracy += test_accuracy
        total_test_mIoU += calculate_iou(test_mask.numpy().squeeze(), (test_predictions.numpy().squeeze() > 0.5), num_classes)
        total_test_mPixIoU += calculate_pix_iou(test_mask.numpy().squeeze(), (test_predictions.numpy().squeeze() > 0.5))
        total_test_mAP += calculate_map(test_mask.numpy().squeeze(), (test_predictions.numpy().squeeze() > 0.5), num_classes)
        num_test_batches += 1

    average_test_loss = total_test_loss / num_test_batches
    average_test_accuracy = total_test_accuracy / num_test_batches
    average_test_mIoU = total_test_mIoU / num_test_batches
    average_test_mPixIoU = total_test_mPixIoU / num_test_batches
    average_test_mAP = total_test_mAP / num_test_batches
    
    print(f"Testing after all epochs and folders: Average Loss: {average_test_loss}, Average Accuracy: {average_test_accuracy}, Test mIoU: {average_test_mIoU}, Test mPixIoU: {average_test_mPixIoU}, Test mAP: {average_test_mAP}")

    # Optionally, you can also print or store the final results for further analysis
    print("Final Results:")
    print(f"Train: Loss: {train_losses}, Accuracy: {train_accuracies}, mIoU: {mIoU_list}, mPixIoU: {mPixIoU_list}, mAP: {mAP_list}")
    print(f"Validation: Loss: {val_losses}, Accuracy: {val_accuracies}, mIoU: {mIoU_list}, mPixIoU: {mPixIoU_list}, mAP: {mAP_list}")
    print(f"Test: Loss: {average_test_loss}, Accuracy: {average_test_accuracy}, mIoU: {average_test_mIoU}, mPixIoU: {average_test_mPixIoU}, mAP: {average_test_mAP}")
    
if __name__ == "__main__":
    main()  