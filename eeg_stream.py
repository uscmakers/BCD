#!/usr/bin/env python3
"""
Read Mindflex CSV lines over Bluetooth SPP and print a tiny dashboard.
Usage:  python3 eeg_stream.py /dev/cu.ESP32_EEG-SerialPort
"""
import sys, time, statistics, collections
from serial import Serial

LABELS = ["signal","attention","meditation",
          "delta","theta","alphaL","alphaH",
          "betaL","betaH","gammaL","gammaM"]

def parse_csv(line: str):
    vals = [int(x.strip()) for x in line.split(",")]
    return dict(zip(LABELS, vals))

def main(port: str):
    ser = Serial(port, 115200, timeout=2)
    print(f"Connected to {port}")
    att_hist = collections.deque(maxlen=10)   # rolling avg
    med_hist = collections.deque(maxlen=10)

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue
            data = parse_csv(line)
            att_hist.append(data["attention"])
            med_hist.append(data["meditation"])

            print("\rSig:{:3}  Att:{:3} (μ:{:4.1f})  Med:{:3} (μ:{:4.1f})"
                  .format(data["signal"],
                          data["attention"], statistics.mean(att_hist),
                          data["meditation"], statistics.mean(med_hist)),
                  end="", flush=True)
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print("\nParse error:", e, file=sys.stderr)
    ser.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: eeg_stream.py <serial-device>")
    main(sys.argv[1])

