""" Generic utilities used for inference"""
import cv2
from inference.constants import XMIN_COORD, YMIN_COORD


def draw_caption(draw, caption, bounding_box):
    """
    Draws given caption on the edge of a bounding box
    :param draw: cv2 draw
    :param caption: string
    :param bounding_box: np.array representing bounding box with xmin, ymin,
    xmax, ymax format
    :return:
    """
    cv2.putText(draw, caption, (bounding_box[XMIN_COORD],
                                bounding_box[YMIN_COORD] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)

    cv2.putText(draw, caption, (bounding_box[XMIN_COORD],
                                bounding_box[YMIN_COORD] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
