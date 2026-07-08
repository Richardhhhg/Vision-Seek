# Tests for the post processor module
# 1. Testing each post processing step individually
# a. Test bw_to_color step
# b. Test stitch step
# 2. Testing the postprocess_video method with multiple steps
# 3. Testing the postprocess_video method with no steps in the preprocessed video
import cv2
import pytest
from detection.data import DetectionModelOutput, PreprocessedVideo
from detection.postprocessing import PostProcessor


@pytest.fixture(scope="module")
def setup():
    input_video_path = "data/test_data/test_video.mp4"
    input_video = cv2.VideoCapture(input_video_path)
    frames = []
    while True:
        ret, frame = input_video.read()
        if not ret:
            break
        frames.append(frame)
    input_video.release()
    preprocessed_video = PreprocessedVideo(frames=frames, steps=[], num_tiles=None, tile_offset=None)
    detection_model_output = DetectionModelOutput(annotated_frames=[], output_video_path="fake_path", preprocessed_video=preprocessed_video)
    post_processor = PostProcessor()
    return post_processor, detection_model_output

def test_bw_to_color_smoke(setup):
    post_processor, detection_model_output = setup
    detection_model_output.preprocessed_video.steps.append("bw")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, DetectionModelOutput)

def test_stitch_smoke(setup):
    # NOTE: this might break if the original input video is a single tile but you try to stitch it as if it wasn't
    post_processor, detection_model_output = setup
    detection_model_output.preprocessed_video.steps.append("resize")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, DetectionModelOutput)

def test_postprocess_video_bw_color_and_stitch_smoke(setup):
    post_processor, detection_model_output = setup
    detection_model_output.preprocessed_video.steps.append("bw")
    detection_model_output.preprocessed_video.steps.append("resize")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, DetectionModelOutput)

def test_postprocess_video_no_steps_smoke(setup):
    post_processor, detection_model_output = setup
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, DetectionModelOutput)