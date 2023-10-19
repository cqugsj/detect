# mqtt_utils.py
import paho.mqtt.client as mqtt

MQTT_BROKER = '101.42.0.63'
MQTT_PORT = 1883
MQTT_USERNAME = 'aisisc2504A'
MQTT_PASSWORD = 'cqdx2504A'
MQTT_TOPIC = 'shiyou/'

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload}")

def setup_mqtt_client():
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()

    return mqtt_client