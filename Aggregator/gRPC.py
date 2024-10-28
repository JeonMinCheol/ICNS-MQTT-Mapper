import grpc
from concurrent import futures
import protos.message_pb2_grpc as pb2_grpc
import protos.message_pb2 as pb2
from aggragator import *
from broker import *

# uplink
# 디바이스 -> 에지 메세지 전달 시 사용 (디바이스에서 사용)
class gRPCClient(object):
    def __init__(self, host, server_port, MAX_MESSAGE_LENGTH  = 100 * 1024 * 1024):
        self.host = host
        self.server_port = server_port
        self.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH # 설정할 최대 메시지 크기 (예: 100MB)
        
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port),
            options=[
                ('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)
            ])

        # 스텁 생성
        self.stub = pb2_grpc.sendParamsStub(self.channel)

    def request(self, message):
        message = pb2.RequestMessage(**message)
        return self.stub.uplink(message)

# server response
class sendParamsService(pb2_grpc.sendParamsServicer):
    def __init__(self, *args, **kwargs):
        self.aggregator = args[0]

    def uplink(self, request, context):
        model, address = request.model, request.address # 들어오면 자동으로 역직렬화 io.bytes
        model = pickle.loads(model) 
        status = {"status": "200", "timestamp" : str(time.time())}
        
        try:
            if self.aggregator.isStillAggregation() or len(self.aggregator.getBuffer()) > self.aggregator.getInput():
                print("Message was ignored cause of aggregation process.")
                status['status'] = "300"
                return pb2.ResponseMessage(**status)
            
            self.aggregator.setBuffer(address, model)
            self.aggregator.aggregation()
            print("request has been successfully handled.")
            return pb2.ResponseMessage(**status)
        except Exception as e:
            print("error", e)
            status['status'] = "500"
            return pb2.ResponseMessage(**status)

    # gRPC server
    def serve(self, MAX_MESSAGE_LENGTH, PORT, aggregator):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(aggregator.getInput()) + 1),
                            options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        pb2_grpc.add_sendParamsServicer_to_server(sendParamsService(aggregator), server)
        server.add_insecure_port(f'[::]:{PORT}')
        server.start()
        print("server start.")
        server.wait_for_termination()