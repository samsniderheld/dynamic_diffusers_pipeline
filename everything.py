from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline
)
from diffusers import UniPCMultistepScheduler

from compel import Compel

import json
from PIL import Image
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

import gradio as gr


"""
this is script that creates a stable diffusion based pipeline via a json file that looks like this
{
    "pipelines": [
      {
        "type": "text2img",
        "model": "SG161222/Realistic_Vision_V1.4",
        "control_net":{
            "model": "lllyasviel/sd-controlnet-canny",
            "interface": {
                "inputs": [
                    "text"
                ]
            }
        },
        "interface": {
            "inputs": [
                "prompt",
                "negative_prompt",
                "image",
                "controlnet_strength",
                "cfg",
                "steps"

            ],
            "outputs": [
                "image"
            ]
        }
        }
    ]
}
"""

def get_controlnet_pipe(pipeline_conf):
    controlnet = ControlNetModel.from_pretrained(pipeline_conf['control_net']['model'])
    return controlnet


def get_pipe(pipeline_conf):
    if pipeline_conf['control_net']:
        controlnet = get_controlnet_pipe(pipeline_conf)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pipeline_conf['model'],
            controlnet=controlnet,
            safety_checker=None,
        ).to('cuda')

    else:
      pipe = StableDiffusionPipeline.from_pretrained(
            pipeline_conf['model'],
            safety_checker=None,
        ).to('cuda')

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    return pipe, compel_proc

with open('config.json', 'r') as file:
    pipelines = json.load(file)

for pipeline in pipelines['pipelines']:
  pipe = pipeline.items()
  for key, value in pipe:
    print(key, value)
  pipe, compel_proc = get_pipe(pipeline)

def create_pipeline_function(pipeline,interface_config):

  def pipeline_function(*args):
    
    prompt = args[0] if "prompt" in interface_config else ""
    negative_prompt = args[1] if "negative_prompt" in interface_config else ""
    init_img = np.array(args[2]) if "image" in interface_config else np.zeros((512,512))
    controlnet_strength = args[3] if "controlnet_strength" in interface_config else .8
    cfg = args[4] if "cfg" in interface_config else 3.5
    steps = args[5] if "csteps" in interface_config else 20

    low_threshold = 100
    high_threshold = 200

    init_img = cv2.resize(init_img,(512,512))

    image = cv2.Canny(init_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    image = Image.fromarray(image)

    random_seed = random.randrange(0,10000)

    texture_image = pipeline(prompt = prompt,
                    negative_prompt = negative_prompt,
                    image= image,
                    controlnet_conditioning_scale=controlnet_strength,
                    height=512,
                    width=512,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]



    return texture_image

  return pipeline_function


def create_gradio_interface(pipeline, pipeline_conf):
  inputs = []
  outputs = []
  if(pipeline_conf["interface"]):
    interface = pipeline_conf["interface"]
  
  if(interface["inputs"]):
    for input in interface["inputs"]:
      print(input)
      if(input == "prompt"):
        prompt_input = gr.Textbox(label="Prompt")
        inputs.append(prompt_input)

      if(input == "negative_prompt"):
        negative_prompt_input = gr.Textbox(label="Negative Prompt")
        inputs.append(negative_prompt_input)

      if(input == "image"):
        image_input = gr.Image(label="Image")
        inputs.append(image_input)

      if(input == "controlnet_strength"):
        controlnet_strength_input = gr.Slider(0.1, 1, value=0.8, label="Controlnet Strength")
        inputs.append(controlnet_strength_input)

      if(input == "cfg"):
        cfg_input = gr.Slider(1, 10, value=3.5, label="CFG")
        inputs.append(cfg_input)

      if(input == "steps"):
        steps_input = gr.Slider(1, 100, value=20, label="Steps")
        inputs.append(steps_input)


  if(interface["outputs"]):
    for output in interface["outputs"]:
      if(output == "text"):
        text_output = gr.Textbox(label="Prompt")
        outputs.append(text_output)
      if(output == "image"):
        image_output = gr.Image(label="Image")
        outputs.append(image_output)

  
  pipeline_function = create_pipeline_function(pipeline, interface['inputs'])

  demo = gr.Interface(fn=pipeline_function, inputs=inputs, outputs=outputs)
  return demo

interface = create_gradio_interface(pipe,pipelines['pipelines'][0])

interface.launch(share=True,debug=True)