import concurrent.futures
import multiprocessing
import wave
import time
from multiprocessing import Manager
import numpy as np
from deepspeech import Model


def transcribe(audio_file, model_path, lm):
    model = Model(model_path)
    model.enableExternalScorer(lm)
    print(f'{ multiprocessing.current_process().name} (Process id: {multiprocessing.current_process().pid}) started.' )
    start = time.time()
    with wave.open(audio_file, "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    text = model.stt(data16)
    end = time.time()
    print(f'{multiprocessing.current_process().name} complete in {round(end - start, 2)} second(s).')
    return text


def speech2text(speech_q, model_path, lm, res):
    while not speech_q.empty():
        model = Model(model_path)
        model.enableExternalScorer(lm)
        audio_file = speech_q.get()
        with wave.open(audio_file, "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            buffer = w.readframes(frames)
        data16 = np.frombuffer(buffer, dtype=np.int16)
        text = model.stt(data16)
        print(text)
        res[audio_file] = text
        return res


if __name__ == '__main__':
    model_path = 'DeepSpeech/deepspeech-0.9.3-models.pbmm'
    lm_path = 'DeepSpeech/deepspeech-0.9.3-models.scorer'
    audio1 = 'DeepSpeech/audio/hello-test.wav'
    audio2 = 'DeepSpeech/audio/4507-16021-0012.wav'
    audio3 = 'DeepSpeech/audio/8455-210777-0068.wav'

    start = time.time()
    with Manager() as manager:
        l = manager.list()
        l.append(audio1)
        l.append(audio2)
        l.append(audio3)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(speech2text, l)
            res = [executor.submit(transcribe, audio, model_path, lm_path) for audio in l]
            for index, f in enumerate(concurrent.futures.as_completed(res)):
                print(f'Audio File {l[index]} is:')
                print(f'    ›{f.result()}.')

        end = time.time()
        print(f'Finished in {round(end - start, 2)} second(s)')
