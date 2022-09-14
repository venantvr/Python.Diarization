"""General helper functions for audio processing"""

import io

import numpy as np
import pydub
import streamlit as st
import streamlit.uploaded_file_manager
from scipy.io import wavfile


def create_st_audio_virtualfile(audio_data: np.array, sample_rate: int) -> io.BytesIO:
    """
    Function that creates the audi virtual file needed to run the audio player
    Args:
        audio_data (np.array): the audio data array
        sample_rate (int): the sampling frequency fo the audio
    Returns:
        io.Bytesio: the audio virtual file stored in temporary memory
    """
    virtual_audio_file = io.BytesIO()
    wavfile.write(virtual_audio_file, rate=sample_rate, data=audio_data)

    return virtual_audio_file


def parse_uploaded_audio_file(uploaded_file: st.uploaded_file_manager.UploadedFile):
    """
    Function that parses the uploaded file into it's a numpy array representation and sampling
    frequency
    Args:
        uploaded_file (st.uploaded_file_manager.UploadedFile): the uploaded streamlit audio file
    """
    audio_data = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = audio_data.split_to_mono()
    samples = [channel.get_array_of_samples() for channel in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], audio_data.frame_rate
