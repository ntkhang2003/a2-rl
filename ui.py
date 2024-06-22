from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import tensorflow as tf

import gradio as gr

import skimage.io as io
from PIL import Image

import network
from actions import command2action, generate_bbox, crop_input
tf.compat.v1.disable_eager_execution()

global_dtype = tf.float32

with open('vfn_rl.pkl', 'rb') as f:
    var_dict = pickle.load(f)

image_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,227,227,3])
global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

h_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,1024])
c_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,1024])
action, h, c = network.vfn_rl(image_placeholder, var_dict, global_feature=global_feature_placeholder,
                                                           h=h_placeholder, c=c_placeholder)
sess = tf.compat.v1.Session()

def auto_cropping(origin_image):
    batch_size = len(origin_image)

    terminals = np.zeros(batch_size)
    ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)
    img = crop_input(origin_image, generate_bbox(origin_image, ratios))

    global_feature = sess.run(global_feature_placeholder, feed_dict={image_placeholder: img})
    h_np = np.zeros([batch_size, 1024])
    c_np = np.zeros([batch_size, 1024])

    while True:
        action_np, h_np, c_np = sess.run((action, h, c), feed_dict={image_placeholder: img,
                                                                    global_feature_placeholder: global_feature,
                                                                    h_placeholder: h_np,
                                                                    c_placeholder: c_np})
        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)
        if np.sum(terminals) == batch_size:
            return bbox
        
        img = crop_input(origin_image, bbox)

def process_img(img):
    im = io.imread(img)
    im_f = im.astype(np.float32) / 255
    xmin, ymin, xmax, ymax = auto_cropping([im_f - 0.5])[0]
    im_crop = im[ymin:ymax, xmin:xmax]
    return im_crop

demo = gr.Interface(
    fn=process_img,
    inputs=gr.Image(label='input', type='filepath'),
    outputs='image',
    theme='shivi/calm_seafoam',
    title='A2-RL',
    css='footer {visibility: hidden}',
    head='<link rel="icon" href="https://upload.wikimedia.org/wikipedia/commons/0/00/Logo_UIT_updated.svg">'
)

if __name__ == '__main__':
    demo.launch()

