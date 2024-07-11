from kafka import KafkaProducer
from json import dumps
import time

producer = KafkaProducer(
    acks=0, # 메시지 전송 완료에 대한 체크
    compression_type='gzip', # 메시지를 전달할 때 압축
    bootstrap_servers=['localhost:9092'], # 전달하고자 하는 카프카 브로커 주소 리스트
    value_serializer=lambda m: dumps(m).encode('utf-8') # 메시지 값 직렬화
)

start = time.time()

for i in range(1):
    data = {
        'music_id': 1234,
        'confidence': 0.90,
        'path': 'C:\\Users\\chang\\Desktop\\음악분석테스트\\',
        'filename': '흔들리는 꽃들 속에서 네 샴푸향이 느껴진거야.mp3'
    }
    producer.send('topic1', value=data)
    producer.flush()

print('[Done]: ', time.time() - start)