import threading
from gRPC import *
from aggragator import *
from pickle import *
from uuid import *
from broker import *
 
MAX_MESSAGE_LENGTH = 1000 * 1024 * 1024
GRPC_CORE_PORT = 50052
GRPC_EDGE_PORT = 50051
KAFKA_BROKER_PORT = 9092
HOST = 'localhost'
TOPIC = "edge"


if __name__ == '__main__':
    client = gRPCClient(HOST, GRPC_CORE_PORT, MAX_MESSAGE_LENGTH)
    broker = Broker(TOPIC, HOST, KAFKA_BROKER_PORT)
    aggregator = Aggregator(broker, client)
    
    t = threading.Thread(target=broker.K_E2D, daemon=True)
    t.start()
    sendParamsService.serve(MAX_MESSAGE_LENGTH, GRPC_EDGE_PORT, aggregator)