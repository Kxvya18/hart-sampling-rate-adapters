# HART — Nicla Vision IMU streamer
# Sends ax,ay,az,gx,gy,gz to RPi over UDP at 30Hz

import network, time, socket
from machine import Pin, SPI, LED
from lsm6dsox import LSM6DSOX

# ── Config — edit these ──────────────────────────────────
SSID    = "Oneplus Nord"           # your WiFi SSID
KEY     = "88888888"      # your WiFi password
RPI_IP  = "10.34.219.50"       # your RPi IP from earlier output
PORT    = 5005
RATE_MS = 33                    # 33ms = ~30Hz
# ─────────────────────────────────────────────────────────

red_led   = LED("LED_RED")
green_led = LED("LED_GREEN")

# ── Connect to WiFi ───────────────────────────────────────
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, KEY)

timeout = 10
while not wlan.isconnected() and timeout > 0:
    print(f'Connecting to "{SSID}"...')
    time.sleep_ms(1000)
    timeout -= 1

if not wlan.isconnected():
    print("❌ WiFi failed")
    red_led.on()
    while True:
        pass

print(f"✅ WiFi connected: {wlan.ifconfig()}")
green_led.on()

# ── UDP socket ────────────────────────────────────────────
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── IMU setup ─────────────────────────────────────────────
spi = SPI(5)
cs  = Pin("PF6", Pin.OUT_PP, Pin.PULL_UP)
lsm = LSM6DSOX(spi, cs)

print(f"Streaming to {RPI_IP}:{PORT} at ~30Hz")
print("Format: ax,ay,az,gx,gy,gz")

# ── Stream loop ───────────────────────────────────────────
while True:
    a = lsm.accel()
    g = lsm.gyro()

    # Comma separated — no timestamp, RPi timestamps on arrival
    data = "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
        a[0], a[1], a[2],
        g[0], g[1], g[2]
    )

    client.sendto(data.encode(), (RPI_IP, PORT))
    time.sleep_ms(RATE_MS)
