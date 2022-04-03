import os
import wave
import time
from multiprocessing import Queue, Process
import numpy as np
from deepspeech import Model


def read_wav_file(filename):
    with wave.open(filename, "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        # print(rate)
        # print(frames)
    return buffer, rate


def transcribe(audio_file, model):
    buffer, rate = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    text = model.stt(data16)
    print(text)
    return text


def speech2text(model, speech_q):
    processes = []
    num_processes = os.cpu_count()
    start = time.time()
    while not speech_q.empty():
        for i in range(num_processes):
            if speech_q.empty():
                break
            audio = speech_q.get(i)
            process = Process(target=transcribe, args=(audio, model,))
            processes.append(process)

        # start all processes
        for process in processes:
            process.start()
            print(f'{process.name} starts')

        for process in processes:
            process.join()

    end = time.time()
    print(f'Finished in {round(end - start, 2)} second(s)')


# transcribe('DeepSpeech/audio/hello-test.wav')
if __name__ == '__main__':
    model_path = 'DeepSpeech/deepspeech-0.9.3-models.pbmm'
    lm_path = 'DeepSpeech/deepspeech-0.9.3-models.scorer'
    beam_width = 500
    lm_a = 0.93
    lm_b = 1.18

    model = Model(model_path)
    model.enableExternalScorer(lm_path)
    model.setScorerAlphaBeta(lm_a, lm_b)
    model.setBeamWidth(beam_width)
    q = Queue()

    audio1 = 'DeepSpeech/audio/hello-test.wav'
    audio2 = 'DeepSpeech/audio/4507-16021-0012.wav'
    audio3 = 'DeepSpeech/audio/8455-210777-0068.wav'
    q.put(audio1)
    q.put(audio2)
    q.put(audio3)
    speech2text(model, q)

