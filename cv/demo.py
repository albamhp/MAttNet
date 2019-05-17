from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import argparse
import os
import os.path as osp
import numpy as np
import torch  # put this before scipy import
from scipy.misc import imread, imresize
import sys
sys.path.insert(0, '../tools')

from mattnet import MattNet


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def show_attn(img_path, box, attn):
    """
    box : [xywh]
    attn: 49
    """
    img = imread(img_path)
    attn = np.array(attn).reshape(7, 7)
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    roi = img[y:y + h - 1, x:x + w - 1]
    attn = imresize(attn, [h, w])
    plt.imshow(roi)
    plt.imshow(attn, alpha=0.7)

def show_boxes(img_path, boxes, colors, texts=None, masks=None):
    # boxes [[xyxy]]
    img = imread(img_path)
    plt.imshow(img)
    ax = plt.gca()
    for k in range(boxes.shape[0]):
        box = boxes[k]
        xmin, ymin, xmax, ymax = list(box)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[k]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        if texts is not None:
            ax.text(xmin, ymin, texts[k], bbox={'facecolor': color, 'alpha': 0.5})
    # show mask
    if masks is not None:
        for k in range(len(masks)):
            mask = masks[k]
            m = np.zeros((mask.shape[0], mask.shape[1], 3))
            m[:, :, 0] = 0
            m[:, :, 1] = 0
            m[:, :, 2] = 1.
            ax.imshow(np.dstack([m * 255, mask * 255 * 0.4]).astype(np.uint8))

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
    args = parser.parse_args('')

    # MattNet
    mattnet = MattNet(args)

    #query
    query = 'bear'
    
    # image path
    IMAGE_DIR = '../../datasets/DAVIS/JPEGImages/1080p/'
    img_path = osp.join(IMAGE_DIR, query, str(0).zfill(5) + '.jpg')

    # forward image
    img_data = mattnet.forward_image(img_path, nms_thresh=0.3, conf_thresh=0.50)
    # show masks
    plt.rcParams['figure.figsize'] = (10., 8.)
    dets = img_data['dets']
    show_boxes(img_path, xywh_to_xyxy(np.array([det['box'] for det in dets])),
               ['blue'] * len(dets), ['%s(%.2f)' % (det['category_name'], det['score']) for det in dets])

    # comprehend expression
    expr = 'man in black'

    entry = mattnet.comprehend(img_data, expr)

    # visualize
    tokens = expr.split()
    print('sub(%.2f):' % entry['weights'][0],
          ''.join(['(%s,%.2f)' % (tokens[i], s) for i, s in enumerate(entry['sub_attn'])]))
    print('loc(%.2f):' % entry['weights'][1],
          ''.join(['(%s,%.2f)' % (tokens[i], s) for i, s in enumerate(entry['loc_attn'])]))
    print('rel(%.2f):' % entry['weights'][2],
          ''.join(['(%s,%.2f)' % (tokens[i], s) for i, s in enumerate(entry['rel_attn'])]))
    # predict attribute on the predicted object
    print(entry['pred_atts'])
    # show prediction
    plt.rcParams['figure.figsize'] = (12., 8.)
    fig = plt.figure()
    plt.subplot(121)
    show_boxes(img_path, xywh_to_xyxy(np.vstack([entry['pred_box']])), ['blue'], texts=None, masks=[entry['pred_mask']])
    plt.subplot(122)
    show_attn(img_path, entry['pred_box'], entry['sub_grid_attn'])
    plt.savefig()

