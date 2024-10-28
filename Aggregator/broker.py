from kafka import KafkaConsumer, KafkaProducer
from pickle import *
import tensorflow as tf
import my_keras
import pickle
import uuid

class Broker():
    def __init__(self, topic, host, port, group_id = "model-group", offset="latest", auto_commit = True, value_serializer=lambda x: dumps(x), value_deserializer=lambda x: loads(x)):
        self.buffer = []
        self.status = False
        self.topic = topic
        self.SIG_SOF = pickle.dumps("SOF")
        self.SIG_EOF = pickle.dumps("EOF")
        self.keras_model = my_keras.create_keras_model()
        self.consumer = KafkaConsumer(
            self.topic, # 토픽명
            client_id=str(uuid.uuid4()),
            bootstrap_servers=[f'{host}:{port}'], # 카프카 브로커 주소 리스트
            auto_offset_reset=offset, # 오프셋 위치(earliest:가장 처음, latest: 가장 최근)
            enable_auto_commit=auto_commit, # 오프셋 자동 커밋 여부
            group_id=group_id, # 컨슈머 그룹 식별자
            value_deserializer=value_deserializer, # 메시지의 값 역직렬화
            # consumer_timeout_ms=1000
        )
        self.producer = KafkaProducer(
            acks=0, # 메시지 전송 완료에 대한 체크
            compression_type='gzip', # 메시지 전달할 때 압축(None, gzip, snappy, lz4 등)
            bootstrap_servers=[f'{host}:{port}'], # 전달하고자 하는 카프카 브로커의 주소 리스트
            value_serializer=value_serializer, # 메시지의 값 직렬화
        )
    
    def getProducer(self):
        return self.producer
    
    def setStatus(self, bool):
        self.status = bool
    
    def getStatus(self):
        return self.status
    
    # edge -> device message 발행
    def K_E2D(self):
        while True:
            SOF = False
            for message in self.consumer:
                if message.value == self.SIG_SOF:
                    print("SIG_SOF")
                    SOF = True
                
                if message.value == self.SIG_SOF or message.value == self.SIG_EOF or SOF:
                    self.producer.send("device", message.value)
                    self.producer.flush()
                    
                if message.value == self.SIG_EOF:
                    print("SIG_EOF")
                    break
                    
                    
    # device message receive    
    def K_RD(self):
        while True:
            SOF = False
            
            # status == 200인 경우만 
            if self.status:
                received_chunks = []
                
                for message in self.consumer:
                    if message.value == self.SIG_SOF:
                        print("model update signal received.")
                        SOF = True
                    
                    elif message.value != self.SIG_SOF and message.value != self.SIG_EOF and SOF:
                        received_chunks.append(message.value)
                        
                    elif message.value == self.SIG_EOF:
                        global_model = -1
                        
                        try:
                            if len(received_chunks) > 0:
                                received_bytes = b''.join(received_chunks)
                                while global_model != -1:
                                    pass
                                global_model = pickle.loads(received_bytes)
                                print("model loaded.")
                            else:
                                break
                                
                        except Exception as e:
                            print('chunk size error', e)
                            
                        try:
                            self.keras_model.set_weights(global_model)
                            print("model updated.")
                            
                        except Exception as e:
                            print("keras_model.set_weights error.", e)
                            
                        finally:
                            self.status = False
                            received_chunks = []
                            
                        break
                        
   

        