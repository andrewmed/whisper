from time import sleep
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import numpy
import pyaudio
import pyperclip
import rumps
import threading
import torch

fs=16000
class Recorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.frames_per_buffer = 1024

    def start(self):
        self.recording = True
        thread = threading.Thread(target=self._record_impl)
        thread.start()
        print("recording...")

    def _record_impl(self):
        self.recording = True
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=fs,
                        frames_per_buffer=self.frames_per_buffer,
                        input=True)
        self.frames = []
        while self.recording:
            data = stream.read(self.frames_per_buffer)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.recording = False
        sleep(0.05)
        return self.frames

class RecorderApp(rumps.App):
    def __init__(self, model):
        super(RecorderApp, self).__init__("🔵", menu=["Transcript"])
        self.recorder = Recorder()
        print('loading model', model)
        self.model=pipeline(
            "automatic-speech-recognition",
            torch_dtype=torch.float16,
            model=model,
            device="mps",
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )
        print('model loaded')

    @rumps.clicked("Transcript")
    def toggle(self, _):
        if self.recorder.recording:
            frames = self.recorder.stop()
            self.menu["Transcript"].title = "Transcript"
            print("transcribing...")
            audio_data = numpy.frombuffer(b''.join(frames), dtype=numpy.int16)
            audio_data_fp32 = audio_data.astype(numpy.float32) / 32768.0
            text = self.model(
                audio_data_fp32,
                chunk_length_s=30,
                batch_size=24,
                return_timestamps=True,
            )
            print(text['text'].strip())
            pyperclip.copy(text['text'].strip())
            self.title='🔵'
        else:
            self.title='🔴'
            self.recorder.start()
            self.menu["Transcript"].title = "Stop"

if __name__ == "__main__":
    app = RecorderApp("openai/whisper-large-v3")
    app.run()

