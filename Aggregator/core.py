from aggragator import *
from gRPC import *

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
KAFKA_BROKER_PORT = 9092
GRPC_CORE_PORT = 50052
TOPIC = "edge"
HOST = 'localhost'

if __name__ == '__main__':
    broker = Broker(TOPIC, HOST, KAFKA_BROKER_PORT)
    aggregator = Aggregator(broker)
    sendParamsService.serve(MAX_MESSAGE_LENGTH, GRPC_CORE_PORT, aggregator)
    
    