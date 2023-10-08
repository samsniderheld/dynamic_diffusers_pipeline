from PIL import Image
import cv2
import numpy as np
import random
import torch
import gradio as gr
from controlnet_aux.processor import Processor

def create_pipeline_function(pipeline,interface_config):

  def pipeline_function(*args):
    params = dict(zip(interface_config['inputs'],args))

    prompt = params['prompt'] if "prompt" in interface_config['inputs'] else ""
    negative_prompt = params['negative_prompt'] if "negative_prompt" in interface_config['inputs'] else ""
    init_img = params['image'] if "image" in interface_config['inputs'] else np.zeros((512,512))
    controlnet_strength = params['controlnet_strength'] if "controlnet_strength" in interface_config['inputs'] else .8
    cfg = params['cfg'] if "cfg" in interface_config['inputs'] else 3.5
    steps = params['steps'] if "steps" in interface_config['inputs'] else 20
    num_samples = params['num_samples'] if "num_samples" in interface_config['inputs'] else 1

    if("control_net" in interface_config):
      init_img = Image.fromarray(init_img).resize((512,512))

      processor = Processor(interface_config["control_net"]["type"])

      processed_img = processor(init_img)

      out_imgs = []

      for i in range(num_samples):

        random_seed = random.randrange(0,10000)

        output_img = pipeline(prompt = prompt,
                        negative_prompt = negative_prompt,
                        image= processed_img,
                        controlnet_conditioning_scale=controlnet_strength,
                        height=512,
                        width=512,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]

        output = np.hstack([processed_img, output_img])

        out_imgs.append(output)

        output_img.save(f"outputs/{i:04d}.png")

      return out_imgs

    else:

      for i in range(num_samples):

        random_seed = random.randrange(0,10000)

        output_img = pipeline(prompt = prompt,
                        negative_prompt = negative_prompt,
                        height=512,
                        width=512,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]

        output_img.save(f"outputs/{i:04d}.png")

      return output_img

  return pipeline_function


def create_gradio_interface(pipeline, pipeline_conf):
    inputs = []
    outputs = []
    if(pipeline_conf["interface"]):
        interface = pipeline_conf["interface"]

    if(interface["inputs"]):
        for input in interface["inputs"]:
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

            if(input == "num_samples"):
                num_samples_input = gr.Slider(1, 100, value=1, label="num samples")
                inputs.append(num_samples_input)


    if(interface["outputs"]):
        for output in interface["outputs"]:
            if(output == "text"):
                text_output = gr.Textbox(label="Prompt")
                outputs.append(text_output)
            if(output == "image"):
                image_output = gr.Image(label="Image")
                outputs.append(image_output)
            if(output == "gallery"):
                gallery_output = gr.Gallery(label="Image")
                outputs.append(gallery_output)

    pipeline_function = create_pipeline_function(pipeline, interface)

    demo = gr.Interface(fn=pipeline_function, inputs=inputs, outputs=outputs)
    return demo