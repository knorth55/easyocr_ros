#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import easyocr
import matplotlib
import matplotlib.cm
import numpy as np
import os
import rospy
import sys
import threading

# OpenCV import for python3
if os.environ['ROS_PYTHON_VERSION'] == '3':
    import cv2
else:
    sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
    import cv2  # NOQA
    sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

# cv_bridge_python3 import
if os.environ['ROS_PYTHON_VERSION'] == '3':
    from cv_bridge import CvBridge
else:
    ws_python3_paths = [p for p in sys.path if 'devel/lib/python3' in p]
    if len(ws_python3_paths) == 0:
        # search cv_bridge in workspace and append
        ws_python2_paths = [
            p for p in sys.path if 'devel/lib/python2.7' in p]
        for ws_python2_path in ws_python2_paths:
            ws_python3_path = ws_python2_path.replace('python2.7', 'python3')
            if os.path.exists(os.path.join(ws_python3_path, 'cv_bridge')):
                ws_python3_paths.append(ws_python3_path)
        if len(ws_python3_paths) == 0:
            opt_python3_path = '/opt/ros/{}/lib/python3/dist-packages'.format(
                os.getenv('ROS_DISTRO'))
            sys.path = [opt_python3_path] + sys.path
            from cv_bridge import CvBridge
            sys.path.remove(opt_python3_path)
        else:
            sys.path = [ws_python3_paths[0]] + sys.path
            from cv_bridge import CvBridge
            sys.path.remove(ws_python3_paths[0])
    else:
        from cv_bridge import CvBridge

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
            '~languages', ['en'])
        self.duration = rospy.get_param('~visualize_duration', 0.1)
        self.enable_visualization = rospy.get_param(
            '~enable_visualization', True)

        self.bridge = CvBridge()
        gpu = rospy.get_param('~gpu', False)
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

        self.pub_rects = self.advertise(
            '~output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            '~output/class', ClassificationResult, queue_size=1)

        if self.enable_visualization:
            self.lock = threading.Lock()
            self.pub_image = self.advertise(
                '~output/image', Image, queue_size=1)
            self.timer = rospy.Timer(
                rospy.Duration(self.duration), self.visualize_cb)
            self.img = None
            self.header = None
            self.bboxes = None
            self.texts = None
            self.scores = None

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
                width=x_max - x_min, height=y_max - y_min)
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

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.header = msg.header
                self.bboxes = bboxes
                self.texts = texts
                self.scores = scores

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None
                or self.header is None or self.bboxes is None
                or self.texts is None or self.scores is None):
            return

        with self.lock:
            vis_img = self.img.copy()
            header = copy.deepcopy(self.header)
            bboxes = self.bboxes.copy()
            texts = self.texts.copy()
            scores = self.scores.copy()

        # bbox
        cmap = matplotlib.cm.get_cmap('hsv')
        n = max(len(bboxes) - 1, 10)
        for i, (bbox, text, score) in enumerate(
                zip(bboxes, texts, scores)):
            rgba = np.array(cmap(1. * i / n))
            color = rgba[:3] * 255
            label_text = '{}, {:.2f}'.format(text, score)
            p1y = max(bbox[0], 0)
            p1x = max(bbox[1], 0)
            p2y = min(bbox[2], vis_img.shape[0])
            p2x = min(bbox[3], vis_img.shape[1])
            cv2.rectangle(
                vis_img, (p1x, p1y), (p2x, p2y),
                color, thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(
                vis_img, label_text, (p1x, max(p1y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                thickness=2, lineType=cv2.LINE_AA)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'rgb8')
        # BUG: https://answers.ros.org/question/316362/sensor_msgsimage-generates-float-instead-of-int-with-python3/  # NOQA
        vis_msg.step = int(vis_msg.step)
        vis_msg.header = header
        self.pub_image.publish(vis_msg)


if __name__ == '__main__':
    rospy.init_node('easyocr_node')
    node = EasyOCRNode()
    rospy.spin()
