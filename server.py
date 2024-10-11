from kafka import KafkaConsumer

import json
from music_analysis import *
from database import close_mysql_connection
from custom_logger import info

consumer = KafkaConsumer(
    'musicAnalysis', # 토픽명
    bootstrap_servers=['localhost:9092'],                           # 카프카 브로커 주소 리스트
    auto_offset_reset='latest',                                     # 오프셋 위치(earliest: 가장 처음, latest: 가장 최근)
    enable_auto_commit=True,                                        # 오프셋 자동 커밋 여부
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),     # 메시지 값의 역직렬화
    consumer_timeout_ms=1000                                        # 데이터를 기다리는 최대 시간
)

info('[Start] consumer started')

load_model()
info('[Model] SPICE model loaded')

try:
    while True:
        # 메시지를 폴링, 타임아웃을 설정하여 주기적으로 메시지 확인
        messages = consumer.poll(timeout_ms=3000)

        if messages:
            for topic_partition, msgs in messages.items():
                for message in msgs:
                    # print(f'Topic : {message.topic}, Partition : {message.partition}, Offset : {message.offset}, Key : {message.key}, value : {message.value}')

                    request = message.value

                    if request['command'] == 'analysis':
                        # 음악 분석 요청
                        audio = convert_audio_for_model(request['music_id'], request['path'], request['filename'])
                        info(f"[Preprocess] Music({request['music_id']}) audio converted")

                        final_outputs = extract_music_pitch(request['music_id'], request['confidence'], request['path'])
                        info(f"[Model] Music({request['music_id']}) pitch extracted")

                        analysis_music(request['music_id'], final_outputs)
                        info(f"[Process] Music({request['music_id']}) analysis completed")

                        save_pitch_audio(request['music_id'], final_outputs, request['path'])
                        info(f"[Audio] Music({request['music_id']}) output audio save completed")

                    elif request['command'] == 'analysis_offline':
                        # 직접 분석 결과 xlsx 파일 업로드 후 분석만 실행
                        final_outputs = load_pitch(request['music_id'], request['path'])
                        info(f"[Load] Music({request['music_id']}) a pitch loaded")

                        analysis_music(request['music_id'], final_outputs)
                        info(f"[Process] Music({request['music_id']}) analysis completed")

                        save_pitch_audio(request['music_id'], final_outputs, request['path'])
                        info(f"[Audio] Music({request['music_id']}) output audio save completed")

                    elif request['command'] == 'delete' and request['action'] == 'value':
                        # 특정 Pitch 값 제거 요청
                        final_outputs = delete_pitch(request['music_id'], request['time'], request['path'])
                        info(f"[Edit] Music({request['music_id']}) a pitch VALUE deleted")

                        analysis_music(request['music_id'], final_outputs)
                        info(f"[Process] Music({request['music_id']}) re-analysis completed")

                        save_pitch_audio(request['music_id'], final_outputs, request['path'])
                        info(f"[Audio] Music({request['music_id']}) output audio re-save completed")

                    elif request['command'] == 'delete' and request['action'] == 'range':
                        # 특정 Pitch 범위 제거 요청
                        final_outputs = delete_pitch_range(request['music_id'], request['time'], request['range'], request['path'])
                        info(f"[Edit] Music({request['music_id']}) a pitch RANGE deleted")

                        analysis_music(request['music_id'], final_outputs)
                        info(f"[Process] Music({request['music_id']}) re-analysis completed")

                        save_pitch_audio(request['music_id'], final_outputs, request['path'])
                        info(f"[Audio] Music({request['music_id']}) output audio re-save completed")

        # else:
        #    print("메시지 없음, 계속 대기 중...")

except PermissionError:
    info('[PermissionError] File I/O error occurred')
except KeyboardInterrupt:
    info('[Interrupt] consumer stopping')
finally:
    consumer.close()
    close_mysql_connection()
    info('[End] consumer stopped')
