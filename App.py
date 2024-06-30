# --encoding: utf8
import os, sys, json, requests, logging, time
import gradio as gr
import math
import numpy as np

print("Launching...")
def resource_path(path = "") -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    else:
        print(os.path.join(os.path.abspath("."), path))
        return os.path.join(os.path.abspath("."), path)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from deepface import DeepFace

window_desc = "This is service for compare, analyse and recognize faces!"
window_tag = "DeepFace-CompareR"
window_icon_path = "DF-C.ico"
logging.basicConfig(level=logging.INFO)

face2face_dropdown = ["VGG-Face", "Facenet", "DeepFace", "OpenFace"]
face2face_checkboxes = ["cosine", "euclidean", "euclidean_l2"]
mytheme = gr.themes.Soft()
mytheme.set(
    # body_background_fill="linear-gradient(0deg, dark, white)"
)

class DF:
    def __init__(self):
        self.win = gr.TabbedInterface(interface_list=[
            gr.Interface(
                # Callback function when tapped Submit button
                fn=self.face2face_compare,
                # Widgets
                inputs=[
                    gr.File(file_count="single", label="First photo for compare"),
                    gr.File(file_count="single", label="Second photo for compare"),
                    gr.Dropdown(choices=face2face_dropdown, value=face2face_dropdown[0], label="Choose a Model for face recognition"),
                    gr.CheckboxGroup(choices=face2face_checkboxes, value=face2face_checkboxes[0], label="Choose a Metric for measuring similarity")
                    # gr.Radio(choices=face2face_checkboxes, type="value", label="Choose a Metric for measuring similarity"),
                ],
                # Output in type TEXT Evolution#hum0n
                outputs=[gr.Text(label="Output")],
                # Window or browser tab title
                title=window_tag,
                description=window_desc,
            ),
            gr.Interface(
                fn=self.face2fld_compare,
                inputs=[
                    gr.File(label="Choose first face"),
                    #
                    # gr.File(file_count="directory", label="Choose a directory for compare faces"),
                    gr.Textbox(value="./db/", label="Enter a directory path for compare faces", placeholder="Path", lines=1),
                    gr.Dropdown(choices=face2face_dropdown, label="Choose a Model for face recognition"),
                    gr.CheckboxGroup(choices=face2face_checkboxes, value=face2face_checkboxes[0], label="Choose a Metric for measuring similarity"),
                    gr.Slider(minimum=0.05, maximum=10.00, step=0.05, label="Specify a threshold to determine whether a pair represents the same person or different individuals.")

                ],
                outputs=["text", "image"]
            )
        ],
            tab_names=['Compare face2face', 'Compare face with faces in directory'],
            theme=mytheme,
            analytics_enabled=True,
            title="DeepFace-CompareR",
        )

        # Launching the app
        self.win.launch(inbrowser=True, server_name="localhost", server_port=1024)
        # self.win.launch(inbrowser=True, share=True)

    def face2face_compare(self, fp0, fp1, model0, metric0):
        print(fp0, fp1, model0, metric0[0])
        result_verify = DeepFace.verify(img1_path=fp0, img2_path=fp1, model_name=model0, distance_metric=metric0[0])
        # result_final = self.plot_fixed_data((0, result_verify['distance']*100), ("N\\S", "Distance"))
        # f_time = time.strftime("%y.%m.%d")
        # s_time = time.strftime("%H~%M~%S")
        # f_path = f'./diagram_{f_time}-{s_time}.png'
        # result_final.imsave(f_path, result_final.get(plt))

        print(str(result_verify))
        # return gr.Text(value=str(result_verify))

        verify_status = 'Failed'
        if result_verify['verified']: verify_status = 'Success!'

        return gr.Text(value=f"Result: {verify_status}\n\nDistance is: {round(number=float(result_verify['distance']), ndigits=2)}\nThreshold used: {result_verify['threshold']}\nModel name: {result_verify['model']}\nSimilarity metric: {result_verify['similarity_metric']}\n\nTime, used for this operation: {result_verify['time']} seconds")
        # return DeepFace.analyze()


    def face2fld_compare(self, fp0, fp1, model0, metric0, trhold0):
        print(fp0, fp1) 
        if os.path.exists('db/'):
            fp1 = './db/'
        result = DeepFace.find(img_path=fp0, db_path=fp1, model_name=model0, distance_metric=metric0[0], threshold=trhold0)
        print(str(result))
        return gr.Text(value=str(result)), gr.Image(value=fp0, width=100, height=200)
        # return DeepFace.analyze()
    def restart_server_public(self):
        self.__init__()

DF()
