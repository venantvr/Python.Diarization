"""
General streamlit diarization application
"""
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, Union

import librosa
import librosa.display
import matplotlib.figure
import numpy as np
import streamlit as st
import streamlit.uploaded_file_manager
from PIL import Image
from matplotlib import pyplot as plt
from pydub import AudioSegment

import configs
from diarizers import pyannote_diarizer
from utils import audio_utils, text_utils, general_utils, streamlit_utils

plt.rcParams["figure.figsize"] = (10, 5)


def plot_audio_diarization(diarization_figure: Union[plt.gcf, np.array], diarization_name: str,
                           audio_data: np.array,
                           sampling_frequency: int):
    """
    Function that plots the audio along with the different applied diarization techniques
    Args:
        diarization_figure (plt.gcf): the diarization figure to plot
        diarization_name (str): the name of the diarization technique
        audio_data (np.array): the audio numpy array
        sampling_frequency (int): the audio sampling frequency
    """
    col1, col2 = st.columns([3, 5])
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.markdown("<br></br>", unsafe_allow_html=True)

        st.audio(audio_utils.create_st_audio_virtualfile(audio_data, sampling_frequency))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>{diarization_name}</h5>",
            unsafe_allow_html=True,
        )

        if type(diarization_figure) == matplotlib.figure.Figure:
            buf = BytesIO()
            diarization_figure.savefig(buf, format="png")
            st.image(buf)
        else:
            st.image(diarization_figure)
    st.markdown("---")


# noinspection PyShadowingNames
def execute_diarization(file_uploader: st.uploaded_file_manager.UploadedFile, selected_option: any,
                        sample_option_dict: Dict[str, str],
                        diarization_checkbox_dict: Dict[str, bool],
                        session_id: str):
    """
    Function that exectutes the diarization based on the specified files and pipelines
    Args:
        sample_option_dict:
        file_uploader (st.uploaded_file_manager.UploadedFile): the uploaded streamlit audio file
        selected_option (any): the selected option of samples
        Dict[str, str]: a dictionary where the name is the file name (without extension to be listed
        as an option for the user) and the value is the original file name
        diarization_checkbox_dict (Dict[str, bool]): dictionary where the key is the Diarization
        technique name and the value is a boolean indicating whether to apply that technique
        session_id (str): unique id of the user session
    """
    user_folder = os.path.join(configs.UPLOADED_AUDIO_SAMPLES_DIR, session_id)
    Path(user_folder).mkdir(parents=True, exist_ok=True)

    if file_uploader is not None:
        file_name = file_uploader.name
        file_path = os.path.join(user_folder, file_name)
        audio = AudioSegment.from_wav(file_uploader).set_channels(1)
        # slice first 30 seconds (slicing is done by ms)
        # audio = audio[0:1000 * 30]  # RVV
        audio.export(file_path, format='wav')
    else:
        file_name = sample_option_dict[selected_option]
        file_path = os.path.join(configs.AUDIO_SAMPLES_DIR, file_name)

    audio_data, sampling_frequency = librosa.load(file_path)

    nb_pipelines_to_run = sum(pipeline_bool for pipeline_bool in diarization_checkbox_dict.values())
    pipeline_count = 0
    for diarization_idx, (diarization_name, diarization_bool) in \
            enumerate(diarization_checkbox_dict.items()):

        if diarization_bool:
            pipeline_count += 1
            diarizer = pyannote_diarizer.PyannoteDiarizer(file_path)

            if file_uploader is not None:
                with st.spinner(
                        f"Executing {pipeline_count}/{nb_pipelines_to_run} diarization pipelines "
                        f"({diarization_name}). This might take 1-2 minutes..."):
                    diarizer_figure = diarizer.get_diarization_figure()
            else:
                diarizer_figure = Image.open(f"{configs.PRECOMPUTED_DIARIZATION_FIGURE}/"
                                             f"{file_name.rsplit('.')[0]}_{diarization_name}.png")

            plot_audio_diarization(diarizer_figure, diarization_name, audio_data,
                                   sampling_frequency)

    shutil.rmtree(user_folder)


st.set_page_config(
    page_title="📜 Audio diarization visualization 📜",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None,
    }
)

text_utils.intro_container()
# 2.1) Diarization method
text_utils.demo_container()
st.markdown("Choose the Diarization method here:")

diarization_checkbox_dict = {}
for diarization_method in configs.DIARIZATION_METHODS:
    diarization_checkbox_dict[diarization_method] = st.checkbox(diarization_method)

# 2.2) Diarization upload/sample select
st.markdown("(Optional) Upload an audio file here:")
file_uploader = st.file_uploader(
    label="", type=[".wav", ".wave"]
)

sample_option_dict = general_utils.get_dict_of_audio_samples(configs.AUDIO_SAMPLES_DIR)
st.markdown("Or select a sample file here:")
selected_option = st.selectbox(
    label="", options=list(sample_option_dict.keys())
)
st.markdown("---")

# 2.3) Apply specified diarization pipeline
if st.button("Apply"):
    session_id = streamlit_utils.get_session()
    execute_diarization(
        file_uploader=file_uploader,
        selected_option=selected_option,
        sample_option_dict=sample_option_dict,
        diarization_checkbox_dict=diarization_checkbox_dict,
        session_id=session_id
    )

text_utils.conlusion_container()
