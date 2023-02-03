# kasanka

### detections_to_tracks.ipynb

Connect frame by frame detections into movement trajectories across time.

Generates:
- raw-tracks.npy

### get-observation-frame-darkness.ipynb

Calculates the average value of the blue channel in each frame of each cameras complete observation (across that camera's clips)

Generates:
- blue-means.npy

### process-tracks.ipynb
Filters out any non-crossing tracks. Then, for each camera at each date, collects all nessissary information about each crossing track into an 'observation' which is saved and then used by the population-estimation notebook (and others) for further analysis.

Requires (for each camera and date being processed):
- raw-tracks.npy (from detections_to_tracks.ipynb)
- blue-means.npy (from get-observation-frame-darkness.ipynb)

### population-estimation.ipynb
Calculate and plot combined population estimates across (a subset of) cameras for each day and across days.