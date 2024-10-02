import os
HOME = os.getcwd()
import sys
import cv2
import numpy as np
import supervision as sv
print(f'supervision version: {sv.__version__}')

from transformers import AutoProcessor, AutoModelForCausalLM

SOURCE_VIDEO_PATH = f"{HOME}/vlc-record-2024-09-12-17h11m17s-rtsp___192.168.0.51_554_-.avi"
TARGET_VIDEO_PATH = f"{HOME}/line_annotator.avi"

# points = []
points = [sv.Point(x=0, y=1875), sv.Point(x=3824, y=1859)]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append(sv.Point(x, y))
        
        print(f"Point : {points[-1]}")
        
        if len(points) == 2:
            cv2.destroyAllWindows()

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", mouse_callback)

while len(points) < 2:
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) == 2:
    line_zone = sv.LineZone(start=points[0], end=points[1])
    print(f"Line defined: Start={points[0]}, End={points[1]}")
else:
    print("Line not defined")
    sys.exit()

boundingbox_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=3, text_scale=1, text_color=sv.Color.BLACK)

line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4,
        text_thickness=4,
        text_scale=2)

line_zone = sv.LineZone(start=points[0], end=points[1])

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
byte_tracker = sv.ByteTrack()

task_prompt = '<OPEN_VOCABULARY_DETECTION>'
text_input = "the product held in hand"

def run_model(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
        
    inputs = processor(
        text = prompt,
        images = image,
        return_tensors = 'pt').to('cuda', model.dtype)
    
    generated_ids = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs['pixel_values'].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3 # 빔 서치(Beam Search) 알고리즘에서 사용하는 매개변수, 빔의 개수를 의미하며, 숫자가 클수록 더 많은 후보 시퀀스를 동시에 탐색
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answers = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(frame.shape[1], frame.shape[0])
    )
    
    return parsed_answers

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    print(f"Frame {index}")
    
    results = run_model(task_prompt, frame, text_input=text_input)
    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, results, resolution_wh=(frame.shape[1], frame.shape[0]))
    detections.class_id = np.array([0] * len(detections))
    detections.confidence = np.array([1.0] * len(detections))
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {text_input} {confidence:0.2f}"
        for confidence, tracker_id
        in zip(detections.confidence, detections.tracker_id)
    ]
    
    # results = model(frame, verbose=False)[0]
    # detections = sv.Detections.from_ultralytics(results)
    # detections = byte_tracker.update_with_detections(detections)

    annotated_frame = frame.copy()
    # annotated_frame = trace_annotator.annotate(
    #     scene=annotated_frame,
    #     detections=detections)
    annotated_frame = boundingbox_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    line_zone.trigger(detections)

    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)
