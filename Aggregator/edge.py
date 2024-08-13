import grpc
from concurrent import futures
import protos.message_pb2_grpc as pb2_grpc
import protos.message_pb2 as pb2
import pickle
import io
import time
from message import *
from aggragator import *

class sendParamsService(pb2_grpc.sendParamsServicer):
    def __init__(self, *args, **kwargs):
        self.aggregator = args[0]

    def sendParamsToEdge(self, request, context):
        model, device = io.BytesIO(request.model), request.device # 들어오면 자동으로 역직렬화
        model = pickle.load(model) 
        self.aggregator.setBuffer(device, model) 
        self.aggregator.aggregation()
        result = {'time': time.time()}
        
        return pb2.Time(**result)

def serve(MAX_MESSAGE_LENGTH, aggregator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(aggregator.getDeviceLen()) + 1),
                         options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])
    pb2_grpc.add_sendParamsServicer_to_server(sendParamsService(aggregator), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("server start.")
    server.wait_for_termination()


if __name__ == '__main__':
    # 설정할 최대 메시지 크기 (예: 100MB)
    MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
    
    client = sendMessageClient()  
    aggregator = Aggregator(client)
    
    serve(MAX_MESSAGE_LENGTH, aggregator)