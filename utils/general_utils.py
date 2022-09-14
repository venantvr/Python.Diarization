"""
General utility functions
"""
import os
from typing import Dict


def get_dict_of_audio_samples(audio_sample_path: str) -> Dict[str, str]:
    """
    Function that returns the list of available audio samples
    Args:
        audio_sample_path (str): The path to the directory with the audio samples
    Returns:
        Dict[str, str]: a dictionary where the name is the file name (without extension to be listed
        as an option for the user) and the value is the original file name
    """
    audio_sample_dict = {}

    for file in os.listdir(audio_sample_path):
        file_option = os.path.basename(file).rsplit('.')[0]
        audio_sample_dict[file_option] = file

    return audio_sample_dict
