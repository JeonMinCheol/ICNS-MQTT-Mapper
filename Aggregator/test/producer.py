from kafka import KafkaProducer
import time
import pickle

producer = KafkaProducer(
    acks=0, # 메시지 전송 완료에 대한 체크
    compression_type='gzip', # 메시지 전달할 때 압축(None, gzip, snappy, lz4 등)
    bootstrap_servers=['localhost:9092'], # 전달하고자 하는 카프카 브로커의 주소 리스트
    value_serializer=lambda x: pickle.dumps(x) # 메시지의 값 직렬화
)

start = time.time()
producer.send('topic2', value=pickle.dumps("SOF"))
producer.flush() # 

for i in range(1000):
    data = {'str' : 'result'+str(i)}
    producer.send('topic2', value=data)
    producer.flush() # 

producer.send('topic2', value=pickle.dumps("EOF"))
producer.flush() # 
print('[Done]:', time.time() - start)