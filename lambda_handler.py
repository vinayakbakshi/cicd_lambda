from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import cv2
import os
import tempfile
import boto3
import tensorflow as tf
from tifffile import imread
from csbdeep.utils import normalize
from stardist import fill_label_holes, random_label_cmap
from stardist.models import Config2D, StarDist2D
from PIL import Image
from io import BytesIO
import boto3

s3 = boto3.client('s3')

np.random.seed(42)
lbl_cmap = random_label_cmap()

def imreadReshape(image):
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    s3.download_file('your-bucket-name', image, tmp_file.name)
    if ".tif" in image:
        imageRead = imread(tmp_file.name)
        if np.ndim(imageRead) == 2:
            return imageRead
        imageRead = np.array(imageRead)
        imageRead = cv2.resize(imageRead,(768,768))
        return imageRead[:,:,0]
    else:
        imageRead = cv2.imread(tmp_file.name)
        if np.ndim(imageRead) == 2:
            return imageRead
        imageRead = cv2.resize(imageRead,(768,768))
        return imageRead[:,:,0]

def lambda_handler(event, context):
    bucket_name = event['bucket']
    image_key = event['key']
    try: 
        response = s3.get_object(Bucket=bucket_name, Key=image_key)
        X_val = [obj['Key'] for obj in event['Records']]
        Y_val = [x.replace('ImagesFullValidation', 'MasksFullValidation') for x in X_val]

        X_val_images = [imreadReshape(image) for image in X_val]

        n_channel = 1 if X_val_images[0].ndim == 2 else X_val_images[0].shape[-1]
        axis_norm = (0,1)  
        
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        X_val_images = [x/255 for x in X_val_images]
        Y_val_images = [fill_label_holes(y) for y in Y_val_images]

        rng = np.random.RandomState(42)

        n_rays = 32
        use_gpu = True

        grid = (2,2)
        conf = Config2D (
            n_rays       = n_rays,
            grid         = grid,
            use_gpu      = use_gpu,
            n_channel_in = n_channel,
            train_patch_size = (768,768)
        )
        model_loaded = StarDist2D(conf)
        new_model = tf.keras.models.load_model('/path/to/your/model_in_s3.keras')
        model_loaded.keras_model = new_model
        model_loaded.thresholds = {"nms":0.3,"prob":0.5590740124765305}
        prediction_second_list = [model_loaded.predict_instances(x, n_tiles=model_loaded._guess_n_tiles(x), show_tile_progress=False)[0]
                    for x in X_val_images]

        for i, prediction in enumerate(prediction_second_list):
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            cv2.imwrite(tmp_file.name, prediction)
            output_key = f'ResultInference37Images/{X_val[i].split("/")[-1].split(".")[0]}.png'
            s3.upload_file(tmp_file.name, bucket_name, output_key)
            tmp_file.close()

        return {
            'statusCode': 200,
            'body': json.dumps('Inference completed successfully!')
        }