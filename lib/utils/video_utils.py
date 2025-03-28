import imageio
from lib.utils.console_utils import *


def write_video(frames, filename, fps=30):
    """
    Write a list of frames to a video file.
    :param frames: list of frames to write
    :param filename: name of the video file to write
    :param fps: frames per second
    """
    try:
        imageio.mimsave(filename, frames, fps=fps)
    except Exception as e:
        print("Error writing video: " + str(e))
        return False
    return True
