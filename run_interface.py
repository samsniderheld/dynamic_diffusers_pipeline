import json
import os
from load_pipelines import get_pipeline
from create_interface import create_gradio_interface

# Create the output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

with open('config.json', 'r') as file:
    pipelines = json.load(file)

for pipeline in pipelines['pipelines']:
  pipe = pipeline.items()
  for key, value in pipe:
    print(key, value)
  pipe, compel_proc = get_pipeline(pipeline)

interface = create_gradio_interface(pipe,pipelines['pipelines'][0])

interface.launch(share=True,debug=True)