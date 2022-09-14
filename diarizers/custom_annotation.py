# noinspection PyPackageRequirements
import os
from math import floor
from pathlib import Path

from pyannote.core import Annotation
from pydub import AudioSegment


class CustomAnnotation(Annotation):
    def get_raw_tracks(self):
        return self._tracks

    def split_at_timestamps(self, audio_path: str):
        results = "results"
        audio = AudioSegment.from_wav(audio_path)
        for track in self.get_raw_tracks().items():
            tuple_track = track[0]
            tuple_who = track[1]
            keys_who = list(tuple_who.keys())
            if len(keys_who) > 0:
                who: str = str(tuple_who[keys_who[0]])
                start = tuple_track.start * 1000
                end = tuple_track.end * 1000
                end = min(floor(audio.duration_seconds * 1000), floor(end + 1000))
                audio_chunk = audio[start:end]
                chunk = "{0}{1}{2}".format(results, os.sep, Path(audio_path).stem)
                name = "{0}{1}{2}".format(chunk, os.sep, who)
                try:
                    os.makedirs(name)
                except OSError:
                    print("Creation of the directory %s failed" % name)
                else:
                    print("Successfully created the directory %s" % name)
                out_f = "{0}{1}{2}-{3}.wav" \
                    .format(name, os.sep, str(floor(start)).zfill(7), str(floor(end)).zfill(7))
                audio_chunk.export(out_f, format="wav")
