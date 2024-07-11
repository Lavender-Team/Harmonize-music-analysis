from kafka import KafkaConsumer

import json
from music_analysis import load_model, convert_audio_for_model, extract_music_pitch, analysis_music
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
info('[Model] spice model loaded')

try:
    while True:
        # 메시지를 폴링, 타임아웃을 설정하여 주기적으로 메시지 확인
        messages = consumer.poll(timeout_ms=3000)

        if messages:
            for topic_partition, msgs in messages.items():
                for message in msgs:
                    # print(f'Topic : {message.topic}, Partition : {message.partition}, Offset : {message.offset}, Key : {message.key}, value : {message.value}')

                    request = message.value

                    audio = convert_audio_for_model(request['music_id'], request['path'], request['filename'])
                    info(f"[Preprocess] Music({request['music_id']}) audio converted")

                    final_outputs = extract_music_pitch(request['music_id'], request['confidence'], request['path'])
                    info(f"[Model] Music({request['music_id']}) pitch extracted")

                    analysis_music(request['music_id'], final_outputs)
                    info(f"[Model] Music({request['music_id']}) analysis completed")

        # else:
        #    print("메시지 없음, 계속 대기 중...")

except PermissionError:
    info('[PermissionError] File I/O error occurred')
except KeyboardInterrupt:
    info('[Interrupt] consumer stopping')
finally:
    consumer.close()
    close_mysql_connection()
    info('[End] consumer stoped')
