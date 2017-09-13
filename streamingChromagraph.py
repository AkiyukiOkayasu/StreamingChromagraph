import pyaudio
import numpy as np
import scipy.signal
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

# High pass filter
FILTERSIZE = 31
NYQ = RESAMPLEDRATE / 2.0  # ナイキスト周波数
CUTOFF = 180.0 / NYQ
b = scipy.signal.firwin(FILTERSIZE, CUTOFF, pass_zero=False)

# 窓関数
window = np.hanning(RESAMPLEDCHUNK)

# OSC ip/portnumber
IP = '127.0.0.1'
PORT = 8080
oscsender = udp_client.UDPClient(IP, PORT)

# wavwrite用のリスト
frames = []

p = pyaudio.PyAudio()


def callback(in_data, frame_count, time_info, status):
    audioin = np.fromstring(in_data, dtype=np.float32)
    audioin = librosa.resample(audioin, RATE, RESAMPLEDRATE)
    audioin = librosa.effects.harmonic(audioin)
    audioin = scipy.signal.lfilter(b, 1, audioin) * window
    chromagram = librosa.feature.chroma_stft(y=audioin, sr=RESAMPLEDRATE, norm=None)
    chromagram = librosa.util.normalize(chromagram, axis=0, threshold=0.3)
    frames.extend(audioin.tolist())

    # CQT = librosa.cqt(harmonic, sr=RATE, n_bins=NUMCHROMA, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN, hop_length=HOPLENGTH)
    # chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0], bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)
    # chromagram = chroma_map.dot(CQT)
    # chromagram = librosa.util.normalize(chromagram, axis=0)
    # chromagram = np.abs(chromagram)

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
time.sleep(20)
stream.stop_stream()

# wav書き出し
ndout = np.array(frames)
librosa.output.write_wav(path='/Users/akiyuki/Desktop/hoge.wav', y=ndout, sr=RESAMPLEDRATE, norm=True)

p.terminate()
print('stoped')
