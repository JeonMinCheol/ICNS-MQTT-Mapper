from kafka import KafkaConsumer
import pickle
consumer = KafkaConsumer(
    'topic2', # 토픽명
    bootstrap_servers=['localhost:9092'], # 카프카 브로커 주소 리스트
    auto_offset_reset='earliest', # 오프셋 위치(earliest:가장 처음, latest: 가장 최근)
    enable_auto_commit=True, # 오프셋 자동 커밋 여부
    group_id='test-group', # 컨슈머 그룹 식별자
    value_deserializer=lambda x: pickle.loads(x), # 메시지의 값 역직렬화
)
 
print('[Start] get consumer')
SIG_SOF = pickle.dumps("SOF")
SIG_EOF = pickle.dumps("EOF")
for message in consumer:
    if message.value == SIG_SOF:
        print("SIG_SOF")
        SOF = True
        
    elif message.value == SIG_EOF:
        print("SIG_EOF")
        break
 
    print(f'Topic : {message.topic}, Partition : {message.partition}, Offset : {message.offset}, Key : {message.key}, value : {message.value}')
print('[End] get consumer')