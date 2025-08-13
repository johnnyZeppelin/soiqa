# This can be a supplementary textual description backup if the DepictQA is temporarily not appliable.
# generate_text.py
from transformers import AutoProcessor, AutoModelForCausalLM
import pandas as pd
from PIL import Image

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda")
processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")

def generate_description(image_path):
    prompt = "Evaluate the image quality with a comprehensive explanation."
    msgs = [{"role": "user", "content": [prompt, Image.open(image_path)]}]
    res = model.chat(processor, msgs)
    return res

# Apply to all images
