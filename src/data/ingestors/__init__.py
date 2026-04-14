"""Data ingestors for all supported datasets."""

from .asvspoof import ingest_asvspoof2019, ingest_asvspoof2021, ingest_asvspoof5
from .wavefake import ingest_wavefake
from .fakeavceleb import ingest_fakeavceleb
from .music import download_musiccaps_audio, ingest_ai_music
from .audioset import download_audioset_subset, ingest_ai_sfx, ingest_esc50
from .voice import ingest_ljspeech, ingest_generic_audio

__all__ = [
    "ingest_asvspoof2019",
    "ingest_asvspoof2021",
    "ingest_asvspoof5",
    "ingest_wavefake",
    "ingest_fakeavceleb",
    "download_musiccaps_audio",
    "ingest_ai_music",
    "download_audioset_subset",
    "ingest_ai_sfx",
    "ingest_esc50",
    "ingest_ljspeech",
    "ingest_generic_audio",
]
