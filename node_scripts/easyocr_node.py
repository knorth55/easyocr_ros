#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt

from chainercv.visualizations import vis_bbox
from cv_bridge import CvBridge
import easyocr
import numpy as np
import rospy

from jsk_topic_tools import ConnectionBasedTransport

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image


class EasyOCRNode(ConnectionBasedTransport):
    def __init__(self):
        super(EasyOCRNode, self).__init__()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        self.languages = rospy.get_param(
            '~languages', ['ja', 'en'])
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
