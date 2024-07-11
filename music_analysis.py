import tensorflow as tf
import tensorflow_hub as hub

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import wavfile
from pydub import AudioSegment

from database import update_music_analysis

EXPECTED_SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768.0
model = None


def load_model():
    global model
    # Tensorflow 1 모델 사용으로 인한 warning 문고 표시 안함
    tf.get_logger().setLevel('ERROR')

    model = hub.load("https://www.kaggle.com/models/google/spice/TensorFlow1/spice/2")
    return


def convert_audio_for_model(music_id, path, filename):
    audio = AudioSegment.from_file(path + filename)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)

    if not os.path.exists(path + f'{music_id}/'):
        os.makedirs(path + f'{music_id}/')
    audio.export(path + f'{music_id}/converted.wav', format='wav')
    return


def extract_music_pitch(music_id, confidence, path):
    # 오디오 파일 로드
    sample_rate, audio_samples = wavfile.read(path + f'{music_id}/converted.wav', 'rb')

    # 오디오 길이 출력
    duration = len(audio_samples) / sample_rate
    print(f'    Sample rate: {sample_rate} Hz')
    print(f'    Total duration: {int(duration / 60)}m {(duration % 60):.2f}s')
    print(f'    Size of the input: {len(audio_samples)}')

    # 정규화
    audio_samples = audio_samples / float(MAX_ABS_INT16)

    # 모델 입력 및 pitch / uncertainty 텐서 출력
    model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))
    pitch_outputs = model_output["pitch"]
    uncertainty_outputs = model_output["uncertainty"]
    print(f'    Total duration: {(pitch_outputs.get_shape().as_list()[0] * 32 / 1000):.2f}s')

    # 추측의 불확실성을 확실성로 변환
    confidence_outputs = 1.0 - uncertainty_outputs

    # 자료형 변환
    pitch_outputs = [float(x) for x in pitch_outputs]
    confidence_outputs = list(confidence_outputs)

    indices = range(len(pitch_outputs))
    # 확실성이 confidence 미만의 값은 0으로 변환
    confident_pitch_outputs = [(i, p if c >= confidence else 0, c) for i, p, c in
                               zip(indices, pitch_outputs, confidence_outputs)]

    outputs_index, outputs_pitch, outputs_confident = zip(*confident_pitch_outputs)

    # 초 단위 추가, 헤르츠 값으로 변환
    final_outputs = [(i, t, p) for i, t, p in
                     zip(outputs_index, index_to_time(outputs_index), value_to_hertz(outputs_pitch))]

    # xlsx 파일로 출력
    df = pd.DataFrame(final_outputs, columns=['index', 'time', 'pitch_point'])
    df.to_excel(path + f'{music_id}/pitch.xlsx', sheet_name='pitch')

    return final_outputs


def analysis_music(music_id, final_outputs):
    update_music_analysis(music_id)
    return


def index_to_time(values):
    # 한 개의 값이 32ms, 인덱스를 초 단위로 변환
    return [x * 32 / 1000 for x in values]


def output2hz(pitch_output):
    # Constants taken from https://tfhub.dev/google/spice/2
    # SPICE 모델의 출력 값을 Hertz로 변환
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


def value_to_hertz(values):
    # list에 대해 output2hz 적용
    return [0 if x == 0 else output2hz(x) for x in values]


def zero_to_nan(values):
    # 0을 nan으로 변환
    return [float('nan') if x == 0 else x for x in values]
