import gradio as gr
import os
import torch

from model import create_convnext_tiny_model
from timeit import default_timer as timer
from typing import Dict, Tuple


with open("class_names.txt", "r") as f:
  class_names = [food_name.split() for food_name in f.readlines()]

model, transforms = create_convnext_tiny_model(num_classes=101)

model.load_state_dict(torch.load(f="pretrained_convnext_tiny_feature_extractor_food101_20_percent.pth", map_location=torch.device("cpu")))

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    start_time = timer()
    
    img = transforms(img).unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    

    return pred_labels_and_probs, pred_time

title = "FoodVision Big 🍔👁"
description = "An ConvNeXt Tiny feature extractor computer vision model to classify images of food into [101 different classes]"
example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)
