import tensorflow as tf
import numpy as np
import preprocess, mura_pipeline
from Models import DenseNet169
import time

batch_size = 1
img_row, img_col = 224, 224
weightPaths = ["./Saved_Models/Final/Dense169_5.h5", "./Saved_Models/Final/Dense169_7.h5", "./Saved_Models/Final/Dense169_8.h5", "./Saved_Models/Final/Dense169_9.h5"]
train_images, train_labels, valid_images, valid_labels, train_set, valid_set = mura_pipeline.getDataSets(batch_size, img_row, img_col)
valid_studies = np.load('valid_studies_array.npy')
predictions = None

model = DenseNet169.densenet169_model(img_rows=img_row, img_cols=img_col, color_type=3, num_classes=1, dropout_rate=0)

for i in range(len(weightPaths)):
    model.load_weights(weightPaths[i], by_name=True)
    model_predictions = model.predict(valid_set, steps=len(valid_images)//batch_size)
    if(predictions is None):
        predictions = np.reshape(model_predictions, len(valid_images)//batch_size)
    else:
        predictions = np.vstack([predictions, np.reshape(model_predictions, len(valid_images)//batch_size)])

indices = np.array(list(range(len(valid_images))))
correct_studies = 0
for study in valid_studies:
    study_indices = indices[np.core.defchararray.find(valid_images, study[0]) != -1]
    study_predictions = predictions[:, (study_indices)]
    average = np.average(study_predictions)
    if (np.equal(np.around(average), int(study[1]))):
        correct_studies += 1

print("Study Accuracy: %d / %d (%2.1f %%)" % (correct_studies, len(valid_studies), 100 * correct_studies/len(valid_studies)))
