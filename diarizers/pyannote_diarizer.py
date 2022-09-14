"""
pyannote diarizer
"""

import matplotlib.pyplot as plt
# noinspection PyPackageRequirements
from pyannote.audio import Pipeline
# noinspection PyPackageRequirements
from pyannote.core import notebook

from diarizers.custom_annotation import CustomAnnotation
from diarizers.diarizer import Diarizer


class PyannoteDiarizer(Diarizer):
    def __init__(self, audio_path: str):
        """
        Pyannote diarizer class
        Note: pyannote does not currently support defining the number of speakers, this
        functionality might be supported later in the future
        Args:
            audio_path (str): the path to the audio file
        Params:
            diarization (Annotations): the output of the diarization algorithm
        """
        self.audio_path = audio_path
        self.diarization = None

    def diarize_audio(self):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        self.diarization = pipeline({'audio': self.audio_path})

    def get_diarization_figure(self) -> plt.gcf:
        if not self.diarization:
            self.diarize_audio()
        figure, ax = plt.subplots()
        notebook.plot_annotation(self.diarization, ax=ax, time=True, legend=True)
        self.split_at_timestamps()
        return plt.gcf()

    def split_at_timestamps(self):
        self.diarization.__class__ = CustomAnnotation
        self.diarization.split_at_timestamps(self.audio_path)
