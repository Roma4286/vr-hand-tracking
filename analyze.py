import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import dotenv_values

env = dotenv_values('.env')
VIDEO_FILE = env['VIDEO_FILE']
TRACKING_FILE = env['TRACKING_FILE']
VIDEO_LENGTH = float(env['VIDEO_LENGTH'])

SMOOTH_WINDOW = 25
ACTIVITY_WINDOW = 180
ACTIVITY_THRESHOLD = 0.18
MIN_PAUSE = 1
GRAPH_LEFT = 0.05
GRAPH_WIDTH = 0.93
GRAPH_BOTTOM = 0.12
GRAPH_HEIGHT = 0.83


def read_number_list(text):
    parts = text.split(',')
    numbers = []
    for i in range(0, len(parts) - 1, 2):
        numbers.append(float(parts[i] + '.' + parts[i + 1]))
    return numbers


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode='same')


times = []
right_wrist = []
left_wrist = []

with open(TRACKING_FILE, encoding='utf-8') as f:
    first_line = json.loads(f.readline())
    start_time = first_line['timeStampNs']

    for line in f:
        frame = json.loads(line)
        times.append((frame['timeStampNs'] - start_time) / 1e9)
        right = frame['Hand']['rightHand']['HandJointLocations'][1]['p']
        left = frame['Hand']['leftHand']['HandJointLocations'][1]['p']
        right_wrist.append(read_number_list(right)[:3])
        left_wrist.append(read_number_list(left)[:3])

times = np.array(times)
right_wrist = np.array(right_wrist)
left_wrist = np.array(left_wrist)
print('кадров:', len(times))

dt = np.diff(times)
right_speed = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1) / dt
left_speed = np.linalg.norm(np.diff(left_wrist, axis=0), axis=1) / dt
times = times[1:]

right_smooth = moving_average(right_speed, SMOOTH_WINDOW)
left_smooth = moving_average(left_speed, SMOOTH_WINDOW)

activity = moving_average(right_speed, ACTIVITY_WINDOW)
pauses = []
start = None
for i in range(len(times)):
    if activity[i] < ACTIVITY_THRESHOLD and start is None:
        start = times[i]
    if activity[i] >= ACTIVITY_THRESHOLD and start is not None:
        if times[i] - start > MIN_PAUSE:
            pauses.append((start, times[i]))
        start = None

print('пауз (высыпаний):', len(pauses))
for p in pauses:
    print(round(p[0], 1), '-', round(p[1], 1), 'с')

fig = plt.figure(figsize=(16, 3.5), dpi=110)
ax = fig.add_axes([GRAPH_LEFT, GRAPH_BOTTOM, GRAPH_WIDTH, GRAPH_HEIGHT])

for p in pauses:
    ax.axvspan(p[0], p[1], color='pink', alpha=0.6)
ax.axvspan(0, 0, color='pink', alpha=0.6, label='высыпание')

ax.plot(times, right_smooth, color='tab:blue', label='правая рука')
ax.plot(times, left_smooth, color='tab:orange', label='левая рука')

ax.set_xlim(0, VIDEO_LENGTH)
ax.set_ylim(0, max(right_smooth.max(), left_smooth.max()) + 0.1)
ax.set_xlabel('время, с')
ax.set_ylabel('скорость запястья, м/с')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

fig.savefig('timeline.png')

page_config = {
    'VIDEO_FILE': VIDEO_FILE,
    'VIDEO_LENGTH': VIDEO_LENGTH,
    'GRAPH_LEFT': GRAPH_LEFT,
    'GRAPH_WIDTH': GRAPH_WIDTH,
    'GRAPH_BOTTOM': GRAPH_BOTTOM,
    'GRAPH_HEIGHT': GRAPH_HEIGHT,
}
with open('config.js', 'w', encoding='utf-8') as f:
    for name, value in page_config.items():
        f.write(f'var {name} = {json.dumps(value)};\n')

print('готово: timeline.png, config.js')
