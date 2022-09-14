"""
Abstract class for diarization
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Diarizer(ABC):
    """
    Diarizer base class
    """

    @abstractmethod
    def get_diarization_figure(self) -> plt.gcf:
        """
        Function that returns the audio plot with diarization segmentations
        Returns:
            plt.gcf: the diarization plot
        """
