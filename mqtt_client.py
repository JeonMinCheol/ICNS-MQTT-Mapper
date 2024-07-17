import paho.mqtt.client as mqtt
import json
import threading
import time

broker_address = ["163.180.117.244", "163.180.117.58"]
model = ["mqtt-device-244", "mqtt-device-58"]

# subscriber callback
# (switch == 0): OFF
# (switch == 1): ON
def on_message(a, b, message):
        json_string = str(message.payload.decode("utf-8"))
        data = json.loads(json_string)["switch"]
        
        switch = "OFF" if data == "0" else "ON"
        print(switch)

def sub(address, model):
        client = mqtt.Client()
        client.connect(address)
        client.subscribe("mqtt/output/device/%s/delta" %model)
        client.on_message = on_message
        client.loop_forever()



for i in range(len(broker_address)):
        thread = threading.Thread(target=sub, args=(broker_address[i], model[i]))
        thread.daemon = True
        thread.start()

while True:
        pass
