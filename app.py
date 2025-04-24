import gradio as gr
import numpy as np
import os
import shutil
import torch
import pandas as pd

# --------------------------
# <<< INTERFACE FUNCTION >>>
# --------------------------
# write the interface function for backend and frontend 
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


# -----------------------------
# <<< LAUNCHING APPLICATION >>>
# -----------------------------

# launching the app with gradio
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
