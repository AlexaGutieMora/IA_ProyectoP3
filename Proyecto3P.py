# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:46:49 2025

@author: k
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt 
from PIL import Image
import cv2                            
sys.path.append("..")

from utils import label_map_util       # utilidades
from utils import visualization_utils as vis_util  # dibujar


cap = cv2.VideoCapture(0)


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT  = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

if not os.path.exists(PATH_TO_CKPT):
    print("Descargando modelo… esto puede tardar un poco.")
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        if 'frozen_inference_graph.pb' in os.path.basename(file.name):
            tar_file.extract(file, os.getcwd())
    print("Modelo descargado y extraído.")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("Modelo cargado en el grafo de TensorFlow.")

label_map     = label_map_util.load_labelmap(PATH_TO_LABELS)
categories    = label_map_util.convert_label_map_to_categories(
                    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Iniciando la cámara. Pulsa la letra q para salir.")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            if not ret:
                print("No se pudo leer la cámara.")
                break
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor     = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes            = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores           = detection_graph.get_tensor_by_name('detection_scores:0')
            classes          = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections   = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Dibujar resultados sobre la imagen
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Mostrar la imagen
            cv2.imshow('Object Detection (TF API)', cv2.resize(image_np, (800, 600)))

            # Salir con q
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")
