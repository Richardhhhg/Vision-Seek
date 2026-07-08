# Tests to write:
# 1. Every different preprocessing step individually
# 1a. Black and white preprocessing -> should return frames with 3 channels but all 3 channels are the same
# 1b. Resize preprocessing -> should return frames with the specified width and height # NOTE: yolo automatically resizes things rather than applying sliding window
# 2. >1 preprocessing step applied on the same video
# 3. No preprocessing steps -> this should just return the video as np.ndarray