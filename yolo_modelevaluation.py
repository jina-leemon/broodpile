## Code from Ulytralytics website:
## https://docs.ultralytics.com/guides/model-evaluation-insights/#accessing-yolo26-metrics

from ultralytics import YOLO

# Load the model
model = YOLO("/media/AntGate/user/patrick/shared/broodpile_pipeline_output/train/weights/best.pt")

# Run the evaluation
results = model.val(data="/media/AntGate/user/patrick/shared/broodpile_pipeline_output/yolo_input/data.yaml")

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean recall:", results.box.mr)