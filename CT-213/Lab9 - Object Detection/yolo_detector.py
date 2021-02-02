from tensorflow.keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid


class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # Todo: implement object detection logic
        input = self.preprocess_image(image)
        output = self.network.predict(input)
        ball_detection, post1_detection, post2_detection = self.process_yolo_output(output)
        return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image / 255.0
        image = np.reshape(image, (1, 120, 160, 3))
        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        ball_index = np.unravel_index(np.argsort(output[:, :, 0].ravel())[-1:], output[:, :, 0].shape)
        ball_cell = np.array(output)[ball_index[0], ball_index[1], :][0]
        p_ball = sigmoid(ball_cell[0])
        x_ball = (ball_index[1]+sigmoid(ball_cell[1]))*coord_scale
        y_ball = (ball_index[0]+sigmoid(ball_cell[2]))*coord_scale
        w_ball = 640 * 5 * np.exp(ball_cell[3])
        h_ball = 640 * 5 * np.exp(ball_cell[4])
        post_indexes = np.array(np.unravel_index(np.argsort(output[:, :, 5].ravel())[-2:], output[:, :, 5].shape))
        post1_cell = np.array(output)[post_indexes[0, 0], post_indexes[1, 0], :]
        post2_cell = np.array(output)[post_indexes[0, 1], post_indexes[1, 1], :]
        p_post1 = sigmoid(post1_cell[5])
        x_post1 = (post_indexes[1, 0] + sigmoid(post1_cell[6])) * coord_scale
        y_post1 = (post_indexes[0, 0] + sigmoid(post1_cell[7])) * coord_scale
        w_post1 = 640 * 2 * np.exp(post1_cell[8])
        h_post1 = 640 * 5 * np.exp(post1_cell[9])
        p_post2 = sigmoid(post2_cell[5])
        x_post2 = (post_indexes[1, 1] + sigmoid(post2_cell[6])) * coord_scale
        y_post2 = (post_indexes[0, 1] + sigmoid(post2_cell[7])) * coord_scale
        w_post2 = 640 * 2 * np.exp(post2_cell[8])
        h_post2 = 640 * 5 * np.exp(post2_cell[9])
        ball_detection = (p_ball, x_ball, y_ball, w_ball, h_ball)
        post1_detection = (p_post1, x_post1, y_post1, w_post1, h_post1)
        post2_detection = (p_post2, x_post2, y_post2, w_post2, h_post2)
        return ball_detection, post1_detection, post2_detection
