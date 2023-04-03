import tiflash
import json

# load the radar configuration from the file
with open('./IWR6843ISK.cfg', 'r') as f:
    config = json.load(f)

# initialize the radar board
radar = tiflash.Radar(config)


# wait until the NRST button is pressed to start the data acquisition
print('Please press the NRST button to start the data acquisition...')
while not radar.is_device_ready():
    pass

print('NRST button pressed. Starting data acquisition...')
i = 0

# continuously read data from the radar board
while True:
    # read one frame of data from the radar
    data = radar.get_frame()
    print(f"DATA RECEIVED: i={i}.  *****")

    # process the data here...

    # repeat for the next frame
