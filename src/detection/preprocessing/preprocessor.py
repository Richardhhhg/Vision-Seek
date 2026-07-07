import cv2
import numpy as np


class Preprocessor:
    """
    Class for preprocessing video for detection

    Attributes:
    - steps (list): List of preprocessing steps to be applied to the video
    - supported_steps(dict [str, callable]): Dictionary of supported preprocessing steps and their corresponding functions

    Methods: 
    - add_step(step: str): Adds a preprocessing step to the list of steps to be applied to the video
    - preprocess_video(video_path: str): Preprocesses the video at the given path according to the added steps and returns the preprocessed video
    - _make_bw(frame: np.ndarray): Converts the given frame to black and white
    - _resize(frame: np.ndarray, width: int, height: int): Resizes the given frame to the specified width and height
    """
    def __init__(self, steps: list = None, **kwargs):
        self.steps = steps if steps is not None else []

        self.supported_steps = {
            "bw": self._make_bw,
            "resize": self._resize,
        }

    def add_step(self, step: str) -> None:
        """
        Adds a preprocessing step to the list of steps to be applied to the video.
        """
        self.steps.append(step)

    def preprocess_video(self, video_path: str) -> np.ndarray:
        """
        Preprocesses the video at the given path according to the added steps and returns the preprocessed video.
        """
        # TODO: look more into detail about this function. I don't think this works...
        # How thing should work
        # Reads the video as numpy array
        # Iterates through frames in the video
        # for each frame, apply the preprocessing steps defined in self.steps
        # return the preprocessed video as a numpy array
        video = cv2.VideoCapture(video_path)
        preprocessed_frames = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            for step in self.steps:
                if step in self.supported_steps:
                    frame = self.supported_steps[step](frame)
                else:
                    raise ValueError(f"Unsupported preprocessing step: {step}")

            preprocessed_frames.append(frame)
        preprocessed_frames = np.array(preprocessed_frames)
        return preprocessed_frames