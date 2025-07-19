import numpy as np
import os
import soundfile as sf
import threading
import tempfile
import time

from collections import deque
from TTS.api import TTS

from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.audio import AudioIU
from retico_core.text import TextIU

MULTILINGUAL_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

class MultilingualTTSModule(AbstractModule):

    @staticmethod
    def name():
        return "Coqui Multilingual TTS Module"

    @staticmethod
    def description():
        return "A module that synthesizes speech in various languages using Coqui."

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return AudioIU

    LANGUAGE_MAPPING = {
        #! tts --list_models
        "all": (MULTILINGUAL_MODEL, "Brenda Stern"),  # multilingual -- supports ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
        "de": ("tts_models/de/thorsten/vits", None),
        "en": ("tts_models/en/ljspeech/vits", None),
        "es": ("tts_models/es/css10/vits", None),
        "fr": ("tts_models/fr/css10/vits", None),
        "fi": ("tts_models/fi/css10/vits", None),
        "hu": ("tts_models/hu/css10/vits", None),
        "it": ("tts_models/it/mai_female/vits", None),
        "ja": ("tts_models/ja/kokoro/tacotron2-DDC", None),
        "lv": ("tts_models/lv/cv/vits", None),
        "nl": ("tts_models/nl/css10/vits", None),
        "pl": ("tts_models/pl/mai_female/vits", None),
        "pt": ("tts_models/pt/cv/vits", None),
    }

    def __init__(self, language=None, dispatch_on_finish=True, frame_duration=0.2, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 22050
        self.wished_language = language if language in self.LANGUAGE_MAPPING else None
        self.dispatch_on_finish = dispatch_on_finish
        self.frame_duration = frame_duration
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = deque()
        self.audio_pointer = 0

    def current_text(self):
        return " ".join(iu.text for iu in self.current_input)

    def process_update(self, update_message):
        if not update_message:
            return None
        final = False

        for iu, ut in update_message:
            if not self.wished_language and hasattr(iu, "language") and iu.language:
                self.language = iu.language
            else:
                self.language = self.wished_language or "all"
            model, speaker = self.LANGUAGE_MAPPING.get(self.language, self.LANGUAGE_MAPPING["all"])
            self.tts = CoquiTTS(model_name=model, speaker=speaker, language=self.language)
            self.sample_rate = self.tts.sample_rate
            if ut == UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
                if hasattr(iu, "committed") and iu.committed:
                    final = True
            elif ut == UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == UpdateType.COMMIT:
                final = True

            current_text = self.current_text()
            if final or any(p in current_text for p in ".!?"):
                self._latest_text = current_text
                
                self.audio_buffer.clear()
                self.audio_pointer = 0
                
                threading.Thread(
                    target=self._synthesize,
                    args=(current_text,),
                    daemon=True
                ).start()
                self.current_input = []

    def _synthesize(self, text):
        # Generate full WAV bytes
        new_audio = self.tts.synthesize(text)
        # Split into chunks
        frame_bytes = int(self.sample_rate * self.frame_duration) * self.samplewidth
        for i in range(0, len(new_audio), frame_bytes):
            chunk = new_audio[i: i + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk.ljust(frame_bytes, b"\x00")
            self.audio_buffer.append(chunk)
        # When done, leave buffer for thread to exhaust

    def _tts_thread(self):
        while self._tts_thread_active:
            time.sleep(self.frame_duration)
            if self.audio_buffer:
                raw_audio = self.audio_buffer.popleft()
            else:
                raw_audio = b"\x00" * (self.samplewidth * int(self.sample_rate * self.frame_duration))
            iu = self.create_iu(self.latest_input_iu)
            iu.set_audio(raw_audio, 1, self.sample_rate, self.samplewidth)
            um = UpdateMessage.from_iu(iu, UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self.audio_buffer.clear()
        self.audio_pointer = 0
        self._tts_thread_active = True
        threading.Thread(target=self._tts_thread, daemon=True).start()

    def shutdown(self):
        self._tts_thread_active = False

class CoquiTTS:
    """
    A thin wrapper around Coqui’s TTS.api.TTS for text-to-waveform synthesis.
    """
    def __init__(self,
                 model_name: str = None,
                 speaker: str = None,
                 language: str = None,
                 caching: bool = True,
                 tmp_dir: str = "~/.cache/coqui_tts"):
        if language is None:
            return
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.caching = caching
        self.tmp_dir = os.path.expanduser(tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.tts = TTS(model_name=self.model_name)
        self.sample_rate = self.tts.synthesizer.output_sample_rate

    def get_cache_path(self, text: str):
        import hashlib
        key = hashlib.sha256(text.encode("utf8") + self.model_name.encode("utf8")).hexdigest()
        return os.path.join(self.tmp_dir, f"{key}.wav")

    def synthesize(self, text: str) -> bytes:
        cache_path = self.get_cache_path(text)
        if self.caching and os.path.exists(cache_path):
            return open(cache_path, "rb").read()
        if self.model_name == MULTILINGUAL_MODEL:
            wav = self.tts.tts(text, speaker=self.speaker, language=self.language)
        else:
            wav = self.tts.tts(text)
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        int16 = (wav * 32767).astype(np.int16)
        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpf.close()
        sf.write(tmpf.name, int16, self.sample_rate, subtype="PCM_16")
        with open(tmpf.name, "rb") as f:
            data = f.read()
        if self.caching:
            with open(cache_path, "wb") as f:
                f.write(data)
        return data
