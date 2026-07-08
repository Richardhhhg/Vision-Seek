from detection.data import DetectionModelOutput, PostprocessorOutput


class Postprocessor:
    """
    For post processing results of detection to resemble original but with the proper detection results.

    NOTE: I think this implementation is fairly fragile right now, but it does get the work done.
    Attributes:
    - steps (list[str]): List of post processing steps to be applied to the video.
    - pre_to_post_map (dict[str, str]): Mapping for preprocessing function to the post processing function.
    - supported_steps (dict[str, callable]): Dictionary of supported post processing steps and their corresponding functions.

    Methods:
    - add_step(step: str): Adds a post processing step to the list of steps to be applied to the video.
    - postprocess_video(preprocessed_video: PreprocessedVideo): Postprocesses the preprocessed video according to the added steps and returns the postprocessed video.
    
    TODO: List the methods that are supported for post processing
    """
    def __init__(self):
        self.steps = []
        self.pre_to_post_map = {
            "bw": "bw_to_color",
            "resize": "stitch",
        }
        self.supported_steps = {
            "bw_to_color": self._bw_to_color,
            "stitch": self._stitch,
        }

    def add_step(self, step: str) -> None:
        """
        Adds a post processing step to the list of steps to be applied to the video.
        """
        self.steps.append(step)

    def postprocess_video(self, detected_video: DetectionModelOutput) -> PostprocessorOutput:
        """
        Postprocesses the detected video according to the added steps and returns the postprocessed video.
        """
        # TODO: implement this thing
        # This should take in the detect video
        # take in the preprocessed video and steps
        # apply the post processing steps to undo the preprocessing
        # shove everything into a DetectionOutput object and return it
        pass