from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,
                       DiffusionPipeline
)
from diffusers import UniPCMultistepScheduler

from compel import Compel

def get_controlnet_pipeline(model):
    controlnet = ControlNetModel.from_pretrained(model)
    return controlnet


def get_pipeline(pipeline_conf):
    if 'control_net' in pipeline_conf['interface']:
        model = pipeline_conf['interface']['control_net']['model']
        controlnet = get_controlnet_pipeline(model)
        if pipeline_conf["type"] == "img2img":
           pipe = DiffusionPipeline.from_pretrained(
                pipeline_conf['model'],
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                safety_checker=None,
            ).to('cuda')
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pipeline_conf['model'],
                controlnet=controlnet,
                safety_checker=None,
            ).to('cuda')

    else:
        if pipeline_conf["type"] == "img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        pipeline_conf['model'],
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
