import pyaudio
import numpy as np
import librosa
import librosa.display
import time
from pythonosc import osc_message_builder
from pythonosc import udp_client

CHUNK = 32768  # バッファーサイズ
RATE = 48000  # サンプルレート
RESAMPLEDRATE = 12000  # リサンプル後のサンプルレート
RESAMPLEDCHUNK = CHUNK * (RESAMPLEDRATE / RATE)  # リサンプル後のバッファーサイズ
FORMAT = pyaudio.paFloat32
# OSC ip/portnumber
IP = '127.0.0.1'
PORT = 8080
oscsender = udp_client.UDPClient(IP, PORT)


p = pyaudio.PyAudio()


def callback(in_data, frame_count, time_info, status):
    audioin = np.fromstring(in_data, dtype=np.float32)
    harmonic = librosa.effects.harmonic(audioin, margin=3.0)
    CQT = librosa.cqt(harmonic, sr=RATE, n_bins=NUMCHROMA, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)
    chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0], bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)
    print(chroma_map[:, 0])
    chromagram = chroma_map.dot(CQT)
    chromagram = librosa.util.normalize(chromagram, axis=0)
    audioin = librosa.resample(audioin, RATE, RESAMPLEDRATE)

    out_data = harmonic.tobytes()
    return (out_data, pyaudio.paContinue)

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
    output=True,
    stream_callback=callback
)


print('start')
stream.start_stream()
time.sleep(20)
stream.stop_stream()
p.terminate()
print('stoped')
