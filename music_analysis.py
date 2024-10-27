import tensorflow as tf
import tensorflow_hub as hub

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import wavfile
from scipy.io.wavfile import write
from pydub import AudioSegment

from pitch_converter import PitchConverter
from database import update_music_analysis, is_artist_female_only

EXPECTED_SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768.0
model = None

# SPICE 모델을 불러옵니다. (최초 1회 실행)
def load_model():
    global model
    # Tensorflow 1 모델 사용으로 인한 warning 문고 표시 안함
    tf.get_logger().setLevel('ERROR')

    model = hub.load("https://www.kaggle.com/models/google/spice/TensorFlow1/spice/2")
    return


# 오디오 frame rate를 16000으로 바꿔 새로운 wav 파일을 만듭니다.
def convert_audio_for_model(music_id, path, filename):
    audio = AudioSegment.from_file(path + filename)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)

    if not os.path.exists(path + f'{music_id}/'):
        os.makedirs(path + f'{music_id}/')
    audio.export(path + f'{music_id}/converted.wav', format='wav')
    return


# 오디오 파일을 SPICE 모델에 넣어 시간에 따른 음계를 예측한 뒤 xlsx 파일로 저장합니다.
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

    # 소리 크기가 큰 부분 찾아서 mask 만들기 (소리 절대값이 크면 1)
    mask = np.abs(audio_samples) <= 0.05
    silence_mask = np.where(mask, 0, 1) # mask가 true면 0으로 설정, false면 1로 설정

    # silence_mask를 출력값처럼 512개 값을 하나로 합쳐 modified_silence_mask 만들기
    silence_mask = [silence_mask[i:i + 512] for i in range(0, len(silence_mask), 512)]
    modified_silence_mask = []
    for sample in silence_mask:
        if sample.sum() > len(sample) / 2:
            modified_silence_mask.append(1)
        else:
            modified_silence_mask.append(0)
    modified_silence_mask = np.array(modified_silence_mask, dtype=np.int64)

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

    # 소리 크기가 작은 부분을 0으로 만들기
    outputs_pitch = filter_silence(outputs_pitch, modified_silence_mask)

    # 모델의 출력값 8개를 합치기 (과도하게 이상치가 생기는 것을 막기 위함)
    pitch_m8 = merge_pitch_values(outputs_pitch)

    # 자료형 변환
    index_m8 = range(len(pitch_m8))

    # 삭제된 Pitch 값을 저장하기 위한 컬럼
    outputs_deleted_m8 = [None] * len(index_m8)

    # 초 단위 추가, 헤르츠 값으로 변환
    final_outputs = [(i, t, p, d) for i, t, p, d in
                     zip(index_m8, index_to_time(index_m8, 32 * 8), value_to_hertz(pitch_m8), outputs_deleted_m8)]

    # xlsx 파일로 출력
    df = pd.DataFrame(final_outputs, columns=['index', 'time', 'pitch', 'deleted'])
    df.to_excel(path + f'{music_id}/pitch.xlsx', sheet_name='pitch', index=False)

    return final_outputs


# 예측된 음계에서 최고음 최저음 등의 정보를 찾은 뒤 데이터베이스에 저장합니다.
def analysis_music(music_id, final_outputs):
    # 자료형 변환
    times = [output[1] for output in final_outputs]
    pitchs = [output[2] for output in final_outputs]
    times = np.array(times)
    pitchs = np.array(pitchs)

    # 대표값 추출
    highest_pitch = np.max(pitchs)
    lowest_pitch = np.min(pitchs[pitchs != 0])

    is_female_only = is_artist_female_only(music_id)

    # 고음의 범위
    # 남 : E4 (미): 약 329.63 Hz 이상 // 여 : C5 (도): 약 523.25 Hz 이상
    HIGH_THRESHOLD = 523.25 if is_female_only else 329.63
    count_high_pitch = np.count_nonzero(pitchs[pitchs >= HIGH_THRESHOLD])

    # 저음의 범위
    # 남 : A2 (라): 약 110 Hz // 여 : D4 (레): 약 293.66 Hz
    LOW_THRESHOLD = 293.66 if is_female_only else 110.00
    count_low_pitch = np.count_nonzero(pitchs[pitchs <= LOW_THRESHOLD])

    count_all_pitch = np.count_nonzero(pitchs)

    # 고음/저음 비율
    high_pitch_ratio = count_high_pitch / count_all_pitch
    low_pitch_ratio = count_low_pitch / count_all_pitch

    # Pitch 평균
    sum = np.sum(pitchs)
    pitch_avg = sum / count_all_pitch

    pitch_stat = build_stat_string(pitchs, count_all_pitch)

    update_music_analysis(music_id, highest_pitch, lowest_pitch, high_pitch_ratio, low_pitch_ratio, pitch_avg, pitch_stat)
    return

# 예측된 음을 다시 오디오로 바꾸기 (관리자 확인용)
def save_pitch_audio(music_id, final_outputs, path):
    # 자료형 변환
    times = np.array([output[1] for output in final_outputs])
    pitchs = np.array([output[2] for output in final_outputs])

    # 샘플링 레이트 설정
    sampling_rate = 16000

    # 소리 신호를 저장할 배열 생성
    audio_signal = np.array([])

    # 각 pitch에 대해 사인파 생성 및 audio_signal에 추가
    for i in range(len(pitchs)):
        # 현재 pitch와 다음 pitch 사이의 시간 계산
        duration = times[i + 1] - times[i] if i + 1 < len(times) else 1
        # 해당 시간 동안의 샘플 개수
        samples = int(duration * sampling_rate)
        # 사인파 생성: 현재 pitch (Hz)에 해당하는 주파수로
        frequency = pitchs[i]
        t = np.linspace(0, duration, samples, False)
        note = np.sin(2 * np.pi * frequency * t)
        # 생성된 사인파를 audio_signal에 추가
        audio_signal = np.concatenate((audio_signal, note))

    # 정규화
    audio_signal = np.int16((audio_signal / audio_signal.max()) * 32767)

    # 오디오 파일로 저장
    filename = 'output_audio.wav'
    write(path + f'{music_id}/' + filename, sampling_rate, audio_signal)
    return


# 직접 분석 결과 xlsx 파일 업로드 후 분석만 실행
def load_pitch(music_id, path):
    # 저장된 xlsx 파일 DataFrame으로 불러오기
    df = pd.read_excel(path + f'{music_id}/pitch.xlsx')

    # 재분석을 진행하기 위해 튜플 리스트로 바꿔 반환
    final_outputs = [(i, t, p, d) for i, t, p, d in
                     zip(df['index'].tolist(), df['time'].tolist(), df['pitch'].tolist(), df['deleted'].tolist())]
    return final_outputs


# 이상치 Pitch 값 하나를 삭제
def delete_pitch(music_id, time, path):
    # 저장된 xlsx 파일 DataFrame으로 불러오기
    df = pd.read_excel(path + f'{music_id}/pitch.xlsx')
    
    # 삭제할 시간대의 인덱스 찾기
    index = df.index[df['time'] == time].tolist()
    
    # 삭제할 값을 deleted 컬럼으로 옮기고 0으로 바꾸기 및 저장
    if df.at[index[0], 'pitch'] != 0:
        df.at[index[0], 'deleted'] = df.at[index[0], 'pitch']
        df.at[index[0], 'pitch'] = 0
        df.to_excel(path + f'{music_id}/pitch.xlsx', sheet_name='pitch', index=False)

    # 재분석을 진행하기 위해 튜플 리스트로 바꿔 반환
    final_outputs = [(i, t, p, d) for i, t, p, d in
                     zip(df['index'].tolist(), df['time'].tolist(), df['pitch'].tolist(), df['deleted'].tolist())]
    return final_outputs


# 특정 Pitch 범위(위 또는 아래)를 삭제
def delete_pitch_range(music_id, time, range, path):
    # 저장된 xlsx 파일 DataFrame으로 불러오기
    df = pd.read_excel(path + f'{music_id}/pitch.xlsx')

    # 삭제할 시간대의 인덱스 찾기
    index = df.index[df['time'] == time].tolist()

    if range == 'upper':
        # 전체 값을 순회하면서 삭제할 시간대의 값 초과이면 0으로 바꾸기
        target_pitch = df.loc[df['time'] == time, 'pitch'].values[0]

        # 삭제할 값을 deleted 컬럼으로 옮기고 0으로 바꾸기 및 저장
        if target_pitch != 0:
            df.loc[df['pitch'] > target_pitch, 'deleted'] = df.loc[df['pitch'] > target_pitch, 'pitch']
            df.loc[df['pitch'] > target_pitch, 'pitch'] = 0
            df.to_excel(path + f'{music_id}/pitch.xlsx', sheet_name='pitch', index=False)

    elif range == 'lower':
        # 전체 값을 순회하면서 삭제할 시간대의 값 미만이면 0으로 바꾸기
        target_pitch = df.loc[df['time'] == time, 'pitch'].values[0]

        # 삭제할 값을 deleted 컬럼으로 옮기고 0으로 바꾸기 및 저장
        if target_pitch != 0:
            df.loc[df['pitch'] < target_pitch, 'deleted'] = df.loc[df['pitch'] < target_pitch, 'pitch']
            df.loc[df['pitch'] < target_pitch, 'pitch'] = 0
            df.to_excel(path + f'{music_id}/pitch.xlsx', sheet_name='pitch', index=False)

    # 재분석을 진행하기 위해 튜플 리스트로 바꿔 반환
    final_outputs = [(i, t, p, d) for i, t, p, d in
                     zip(df['index'].tolist(), df['time'].tolist(), df['pitch'].tolist(), df['deleted'].tolist())]
    return final_outputs


def index_to_time(values, time_step = 32):
    # 한 개의 값이 32ms, 인덱스를 초 단위로 변환
    return [x * time_step / 1000 for x in values]


def merge_pitch_values(pitch_output):
    # 모델의 출력값 8개를 합치기 (과도하게 이상치가 생기는 것을 막기 위함)
    merged_values = []
    for i in range(0, len(pitch_output), 8):
        segment = pitch_output[i:i+8]
        non_zero_values = [value for value in segment if value != 0]
        if non_zero_values:
            median_value = np.median(non_zero_values)
            merged_values.append(median_value)
        else:
            merged_values.append(0)
    return merged_values


# 소리 크기가 작은 부분을 0으로 만들기
def filter_silence(values, silence_mask):
    if len(values) > len(silence_mask):
        silence_mask = np.pad(silence_mask, (0, len(values) - len(silence_mask)), 'constant', constant_values=0)

    return [x if silence_mask[i] == 1 else 0 for i, x in enumerate(values)]


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

def build_stat_string(pitchs, count_all_pitch):
    stat = dict()

    scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    octave = ['2', '3', '4', '5', '6']

    for ot in octave:
        if ot != '6':
            for s in scale:
                stat[s+ot] = 0
        else:
            for s in scale[0:2]:
                stat[s+ot] = 0

    for p in pitchs:
        if p != 0:
            pitch_string = PitchConverter.freq_to_pitch(p)
            stat[pitch_string] += 1

    for k, v in stat.items():
        stat[k] = v / count_all_pitch

    return stat
