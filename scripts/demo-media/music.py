"""Original background music for the FrameScale AI demo (synthesized, no samples).

120 BPM, downbeats at 1.5 s + 2k s so scene changes (3.5, 9.5, 15.5 s) land on
a bar. Am – F – C – G, pad + arpeggio + bass, drums from the first scene change.
"""
import sys, wave
import numpy as np

OUT, DUR = sys.argv[1], float(sys.argv[2])
SR = 48000
N = int(DUR * SR)
t = np.arange(N) / SR
BEAT, OFF = 0.5, 1.5
L = np.zeros(N); R = np.zeros(N)

def hz(note):  # MIDI → Hz
    return 440.0 * 2 ** ((note - 69) / 12)

CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]), (55, [55, 59, 62])]  # Am F C G
def chord_at(time):
    bar = int(np.floor((time - OFF) / (4 * BEAT)))
    return CHORDS[bar % 4]

def add(sig, start, gain=1.0, pan=0.0):
    i = int(start * SR)
    if i >= N:
        return
    if i < 0:
        sig = sig[-i:]; i = 0
    n = min(len(sig), N - i)
    L[i:i + n] += sig[:n] * gain * (1 - pan) / 2 * 2 ** 0.5
    R[i:i + n] += sig[:n] * gain * (1 + pan) / 2 * 2 ** 0.5

def soft_saw(f, length, harmonics=7, detune=0.0):
    tt = np.arange(int(length * SR)) / SR
    out = np.zeros_like(tt)
    for k in range(1, harmonics + 1):
        out += np.sin(2 * np.pi * f * k * (1 + detune) * tt) / k * np.exp(-0.15 * k)
    return out

# Pad: one chord per bar (and the pickup), slow attack, overlapping release.
bars = np.arange(-1, int((DUR - OFF) / 2) + 2)
for b in bars:
    start = OFF + b * 4 * BEAT
    root, notes = CHORDS[b % 4]
    length = 2.6
    env_t = np.arange(int(length * SR)) / SR
    env = np.minimum(1, env_t / 0.6) * np.minimum(1, (length - env_t) / 0.6)
    for n in notes:
        for det, pan in ((-0.003, -0.5), (0.003, 0.5)):
            add(soft_saw(hz(n), length, 6, det) * env, start, 0.035, pan)

# Arpeggio: eighth notes over the chord, plucky, with a dotted-eighth echo.
arp = np.zeros(N)
k = 0
time = OFF - 2 * BEAT
while time < DUR - 2.5:
    root, notes = chord_at(max(time, OFF))
    seq = [notes[0] + 12, notes[1] + 12, notes[2] + 12, notes[1] + 24]
    f = hz(seq[k % 4])
    ln = 0.45
    tt = np.arange(int(ln * SR)) / SR
    tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 9)
    i = int(time * SR)
    n = min(len(tone), N - i)
    if i >= 0:
        arp[i:i + n] += tone[:n] * (0.5 if time < 3.5 else 1.0)
    time += BEAT / 2; k += 1
echo = np.zeros(N)
d = int(0.375 * SR)
echo[d:] += arp[:-d] * 0.35
echo[2 * d:] += arp[:-2 * d] * 0.12
L += (arp * 0.7 + echo * 0.3) * 0.09
R += (arp * 0.3 + echo * 0.7) * 0.09

# Bass on beats 1 and 3.
time = OFF
while time < DUR - 2:
    root, _ = chord_at(time)
    ln = 0.9
    tt = np.arange(int(ln * SR)) / SR
    tone = np.sin(2 * np.pi * hz(root - 12) * tt) * np.minimum(1, tt / 0.02) * np.exp(-tt * 2.2)
    add(tone, time, 0.16 if time >= 3.5 else 0.08)
    time += 2 * BEAT

# Drums from the first scene change; hats from the second.
rng = np.random.default_rng(7)
time = 3.5
while time < DUR - 4:
    tt = np.arange(int(0.35 * SR)) / SR
    f = 45 + 75 * np.exp(-tt * 30)
    kick = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 9)
    add(kick, time, 0.32)
    if time >= 9.5:
        hn = int(0.06 * SR)
        noise = rng.standard_normal(hn)
        noise = np.diff(noise, prepend=0)  # brighter
        add(noise * np.exp(-np.arange(hn) / SR * 70), time + BEAT / 2, 0.035, 0.3)
    time += BEAT

# Final chord swell for the end card.
for n in [45, 57, 60, 64, 69]:
    ln = DUR - 21.5
    tt = np.arange(int(ln * SR)) / SR
    env = np.minimum(1, tt / 0.3) * np.exp(-tt * 0.5)
    add(soft_saw(hz(n), ln, 5) * env, 21.5, 0.04, 0.0)

# Master: fade in/out, gentle saturation, normalize to -1 dBFS.
fade = np.minimum(1, t / 0.8) * np.minimum(1, (DUR - t) / 1.8)
mix = np.stack([L, R]) * fade
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix /= np.max(np.abs(mix)) / 10 ** (-1 / 20)
pcm = (mix.T * 32767).astype(np.int16)
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", OUT, DUR, "s")
