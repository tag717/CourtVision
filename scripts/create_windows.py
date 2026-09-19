import csv
from pathlib import Path
from collections import defaultdict
from src.constants import SHOT_LABELS, LABEL_NORMALIZATION, SHOT_OUTCOME_MAP

WINDOW_SIZE_SEC = 2.0 # window length = how much video each sample contains
WINDOW_STRIDE_SEC = 1.0 # stride = how far forward we move to create the next sample
MIN_SHOT_OVERLAP_RATIO = 0.5

def load_events(events_path):
    # Read events.csv
    # Ignore invalid intervals
    # Group rows by (video_id, possession_id)
    pass 

def get_video_path(row, videos_root):
    video_filename = Path(row["source_file"]).with_suffix(".mp4").name

    return (
        Path(videos_root)
        / row["video_id"]
        / video_filename
    )

def get_shot_intervals(events):
    # Read events.csv
    # Ignore invalid intervals
    # Group rows by (video_id, possession_id)
    pass

def calculate_overlap(window_start, window_end, shot_start, shot_end):
    overlap = max(
        0.0,
        min(window_end, shot_end) 
        - max (window_start, shot_start)
    )
    pass

