import numpy as np
import torch
import time
import pickle

# list1에 계산을 더해나갈 것이기 때문에 list1은 항상 같은 리스트가 와야함
def average_homogeneous(list1, list2, weight):
    if type(list1) != type(list2):
        print(f"list1 type: {type(list1)} list2 type: {type(list2)}")
        return average_homogeneous(list1, list2[0], weight)
    else:
        shape1, shape2 = list1.shape, list2.shape
        assert(shape1 == shape2)
        
        x = (list1.reshape(-1) + list2.reshape(-1)) * weight
        return x.reshape(shape1)
    
def federated_average(client_parameters, client_sizes):
    weighted_average = client_parameters[0]
    weight = 1 / client_sizes
    dim1 = len(client_parameters[0])
    
    for param in client_parameters[1:]:
        for i in range(dim1):
            weighted_average[i] = average_homogeneous(weighted_average[i], param[i], weight)
                
    return weighted_average

class Aggregator():
    def __init__(self, client):
        self.BUFFER = [] # 파라미터를 저장할 버퍼
        self.DEVICE_BUFFER_LEN = 0 # 현재 버퍼에 존재하는 아이템의 개수, 코어로 파라미터를 전송 후 초기화
        self.DEVICE_LEN = 0  # 입력으로 전달받은 디바이스 개수
        self.DEVICE_LIST = {} # 파라미터를 전달한 디바이스 목록을 유지
        self.client = client
        self.DEFAULT_WAITING_TIME = 5
        self.setDeviceLen()
        self.setDefaultTime(self.DEFAULT_WAITING_TIME)
        
    def setDefaultTime(self, time=20):
        self.DEFAULT_WAITING_TIME = time
    
    def setDeviceLen(self):
        self.DEVICE_LEN = int(input("enter device length: "))
        
    def getDeviceLen(self):
        return self.DEVICE_LEN
    
    def isContainDevice(self, device):
        return self.DEVICE_LIST.__contains__(device)
    
    def setDeviceList(self, device):
        self.DEVICE_LIST[device] = self.DEVICE_BUFFER_LEN
        self.DEVICE_BUFFER_LEN += 1
        
    def getDeviceList(self):
        return self.DEVICE_LIST, self.DEVICE_BUFFER_LEN
    
    def setBuffer(self, device, param):
        if self.isContainDevice(device):
            index = int(self.getDeviceList()[0][device])
            self.getBuffer()[index] = param
        else:
            self.getBuffer().append(param)
            self.setDeviceList(device)
            
    def getBuffer(self):
        return self.BUFFER
    
    def clearList(self):
        self.BUFFER.clear()
        self.DEVICE_LIST.clear()
        self.DEVICE_BUFFER_LEN = 0
        
    # timer
    def isTimeout(self, start):
        return (time.time() - start >= self.DEFAULT_WAITING_TIME)
        
    def aggregation(self):
        if self.getDeviceLen() <= self.DEVICE_BUFFER_LEN: # 연결된 디바이스로부터 파라미터를 전부 전달받은 이후 데이터 전송
            print("aggregation start.")
            average_weights = federated_average(self.getBuffer(), self.getDeviceLen())
            
            # TODO: send parameter to core
            param = pickle.dumps(average_weights) # Type cast: -> bytes
        
            # Send Data (device : ip, model : parameter)
            # result = self.client.get_url({"device":"test", "model" : param})
            print("send data to core.")
            
            self.clearList()
        else:
            print(f"DEVICE_LEN > DEVICE_BUFFER_LEN ({self.getDeviceLen()} > {self.DEVICE_BUFFER_LEN})")