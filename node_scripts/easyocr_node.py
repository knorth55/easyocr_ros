#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt

from chainercv.visualizations.vis_image import vis_image
from cv_bridge import CvBridge
import easyocr
import numpy as np
import rospy

from jsk_topic_tools import ConnectionBasedTransport

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image


def vis_bbox(img, bbox, label=None, score=None, label_names=None,
             instance_colors=None, alpha=1., linewidth=3.,
             sort_by_score=True, ax=None):
    from matplotlib import pyplot as plt

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    if sort_by_score and score is not None:
        order = np.argsort(score)
        bbox = bbox[order]
        score = score[order]
        if label is not None:
            label = label[order]
        if instance_colors is not None:
            instance_colors = np.array(instance_colors)[order]

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append(u"{:.2f}".format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    u": ".join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, },
                    fontsize=5)
    return ax


class EasyOCRNode(ConnectionBasedTransport):
    def __init__(self):
        super(EasyOCRNode, self).__init__()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        self.languages = rospy.get_param(
            '~languages', ['en'])
        self.bridge = CvBridge()
        gpu = rospy.get_param('~gpu', False)
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

        self.pub_rects = self.advertise(
            '~output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            '~output/class', ClassificationResult, queue_size=1)
        self.pub_image = self.advertise(
            '~output/image', Image, queue_size=1)

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            '~input', Image, self.image_cb, queue_size=1, buff_size=2**26)

    def unsubscribe(self):
        self.sub_image.unregister()

    @property
    def visualize(self):
        return self.pub_image.get_num_connections() > 0

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # RGB -> BGR
        rospy.logdebug('start readtext')
        results = self.reader.readtext(img[:, :, ::-1])
        rospy.logdebug('end readtext')

        bboxes = []
        scores = []
        texts = []
        rect_msg = RectArray(header=msg.header)
        for result in results:
            bb = result[0]
            text = result[1]
            score = result[2]
            x_min = int(np.round(bb[0][0]))
            y_min = int(np.round(bb[0][1]))
            x_max = int(np.round(bb[2][0]))
            y_max = int(np.round(bb[2][1]))
            bboxes.append([y_min, x_min, y_max, x_max])
            texts.append(text)
            scores.append(score)
            rect = Rect(
                x=x_min, y=y_min,
                width=x_max-x_min, height=y_max-y_min)
            rect_msg.rects.append(rect)
        bboxes = np.array(bboxes)
        scores = np.array(scores)

        cls_msg = ClassificationResult(
            header=msg.header,
            classifier=self.classifier_name,
            target_names=texts,
            labels=np.arange(len(texts)),
            label_names=texts,
            label_proba=scores)

        self.pub_rects.publish(rect_msg)
        self.pub_class.publish(cls_msg)

        if self.visualize:
            fig = plt.figure(
                tight_layout={'pad': 0})
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            vis_bbox(
                img.transpose((2, 0, 1)),
                bboxes, np.arange(len(texts)), scores,
                label_names=texts, ax=ax)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            vis_img = np.fromstring(
                fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_img.shape = (h, w, 3)
            fig.clf()
            plt.close()
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'rgb8')
            vis_msg.header = msg.header
            self.pub_image.publish(vis_msg)


if __name__ == '__main__':
    rospy.init_node('easyocr_node')
    node = EasyOCRNode()
    rospy.spin()
