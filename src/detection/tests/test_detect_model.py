# Testing the detection model
# 1. pass basic video into the model, make sure it returns a list of detection frames. Ensure all fields in the detection frames are populated correctly. (may have none if no detections)
# 2. Run on cuda vs cpu, ensure that everything still is populated correctly and doesn't crash