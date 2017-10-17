import pyaudio
import numpy as np
import scipy.signal
import librosa
import librosa.display
import time
from pythonosc import osc_message_builder
from pythonosc import udp_client

# Audio Settings
CHUNK = 16384  # バッファーサイズ
RATE = 48000  # サンプルレート
RESAMPLEDRATE = 24000  # リサンプル後のサンプルレート
FORMAT = pyaudio.paFloat32

# OSC Settings
IP = '127.0.0.1'
PORT = 8080
oscsender = udp_client.UDPClient(IP, PORT)

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    audioin = np.fromstring(in_data, dtype=np.float32)
    audioin = librosa.resample(audioin, RATE, RESAMPLEDRATE)
    audioin = librosa.effects.harmonic(audioin)
    chromagram = librosa.feature.chroma_stft(y=audioin, sr=RESAMPLEDRATE, norm=None)
    chromagram = librosa.util.normalize(chromagram, axis=0, threshold=0.3)

    # OSC send
    msg = osc_message_builder.OscMessageBuilder(address='/chromagram')
    for i in range(12):
        msg.add_arg(chromagram[i, 0])
    msg = msg.build()
    oscsender.send(msg)
    return (None, pyaudio.paContinue)


stream = p.open(
    format=FORMAT,
    channels=1,
    rate=RATE,
    frames_per_buffer=CHUNK,
    input=True,
    output=False,
    stream_callback=callback
)


print('start')
stream.start_stream()

time.sleep(20)# 20秒後にストップ

stream.stop_stream()
p.terminate()
print('stoped')
