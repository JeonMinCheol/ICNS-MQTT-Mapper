import time
import socket
import pickle
import uuid
from federate_average import *

class Aggregator():
    def __init__(self, broker, client = -1):
        self.BUFFER = [] # 파라미터를 저장할 버퍼
        self.BUFFER_LENGTH = 0 # 현재 버퍼에 존재하는 아이템의 개수, 코어로 파라미터를 전송 후 초기화
        self.INVENTORY = set() # 파라미터를 전달한 디바이스 목록을 유지, edge의 경우 set 내부 원소의 개수가 입력받은 것과 동일해야함 (수정 필요)
        self.INPUT_LENGTH = 0  # 입력으로 전달받은 디바이스, 에지 개수
        self.uuid = uuid.uuid4()
        self.agg = False
        self.client = client
        self.DEFAULT_WAITING_TIME = 5
        self.broker = broker
        self.producer = broker.getProducer()
        self.SIG_SOF = pickle.dumps("SOF")
        self.SIG_EOF = pickle.dumps("EOF")
        self.setInput()
        self.setDefaultTime(self.DEFAULT_WAITING_TIME)
        
    def setDefaultTime(self, time=20):
        self.DEFAULT_WAITING_TIME = time
    
    def setInput(self):
        if self.client != -1:
            self.INPUT_LENGTH = int(input("Enter number of device : "))
        else:
            self.INPUT_LENGTH = int(input("Enter number of edge : "))
        
    def getInput(self):
        return self.INPUT_LENGTH
    
    def setInventory(self, address):
        self.INVENTORY.add(address)
        
    def getInventory(self):
        return self.INVENTORY, self.BUFFER_LENGTH
    
    # buffer에 데이터 추가하는 메서드
    def setBuffer(self, address, param):
        self.getBuffer().append(param)
        self.setInventory(address)
        self.BUFFER_LENGTH = len(self.BUFFER)
            
    def getBuffer(self):
        return self.BUFFER
    
    def clearList(self):
        self.BUFFER.clear()
        self.INVENTORY.clear()
        self.BUFFER_LENGTH = 0
        
    def isTimeout(self, start):
        return (time.time() - start >= self.DEFAULT_WAITING_TIME)
    
    def makeChunks(self, data, chunk_size = 128 * 1024): # default : 128KB
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks
    
    def isStillAggregation(self):
        return self.agg
        
        
    # TODO : 1. 리스트 방식에서 카운트 방식으로 수정할 것 (완료)
    # TODO : 2. aggregation하는 동안 전달되는 파라미터 버리기 (완료)
    # TODO : 3. 디바이스에서 파라미터 전송한 경우에만 모델 업데이트할 것 (완료, 검증 필요)
    # TODO : 4. 글로벌 모델을 전달받은 디바이스는 새롭게 학습 후 전달할 것 (완료, 검증 필요)
    # TODO : 5. 쓰레드 세이프하게 수정 
    
    def aggregation(self):
        # 연결된 디바이스로부터 파라미터를 전부 전달받은 이후 데이터 전송
        if (self.client != -1 and self.getInput() <= self.BUFFER_LENGTH) or (self.client == -1 and self.getInput() <= len(self.INVENTORY)):
            print("Aggregation start.")
            average_weights = federated_average(self.getBuffer(), self.getInput())
            self.agg = True
            self.clearList() # 초기화
            
            if self.client != -1:
                # 코어로 모델 전송
                try:
                    dump = -1
                    dump = pickle.dumps(average_weights)
                    while dump == -1: # dump 대기 (지우면 에러)
                        pass
                    
                    message = {"address" : socket.gethostbyname(socket.gethostname()) + str(self.uuid), "model" : dump}
                    response = self.client.request(message=message)
                    print(response)
                    
                except Exception as e:
                    print(e)
                    
                else:
                    print("send to core")
            else:
                # global model 반환
                try:
                    dump = -1
                    dump = pickle.dumps(average_weights)
                    while dump == -1: # dump 대기 (지우면 에러)
                        pass
                    
                    chunks = self.makeChunks(dump)
                    print(f"chunk size : {len(chunks)}")
                    
                    s = time.time()
                    
                    # 시작 신호 전송
                    self.producer.send(topic = "edge", value = self.SIG_SOF)
                    
                    for chunk in chunks:
                        self.producer.send(topic = "edge",  value = chunk)
                        self.producer.flush()
                        
                    # 종료 신호 전송
                    self.producer.send(topic = "edge", value = self.SIG_EOF)
                    print(f"send global model. {time.time() - s}/sec")
                except Exception as e:
                    print("send error", e)
                    
            self.clearList() # 초기화
            self.agg = False
                    
            print("Aggregation complete.")
        elif self.client != -1:
            print(f"DEVICE_LEN > BUFFER_LENGTH ({self.getInput()} > {self.BUFFER_LENGTH})")
        else:
            print(f"DEVICE_LEN > BUFFER_LENGTH ({self.getInput()} > {len(self.INVENTORY)})")