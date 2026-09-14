import csv
import random 
from collections import Counter

video_ids = set()

# 1. parse
with open('data/metadata/events.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        video_ids.add(row["video_id"])

# show how many videos we have
print(len(video_ids))
# print(video_ids)  

video_ids = list(video_ids)

# 2. split videos
random.seed(42) 
random.shuffle(video_ids)

train_videos = video_ids[:22]
validation_videos = video_ids[22:27]
test_videos = video_ids[27:]

print("Train:", len(train_videos))
print("Validation:", len(validation_videos))
print("Test:", len(test_videos))

# 3. Create Counters 
train_shots = Counter()
validation_shots = Counter()
test_shots = Counter()

# 4. Count shot types for each split
with open("data/metadata/events.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:

        # Ignore non-shot events
        if row["is_shot"] != "1":
            continue

        # Ignore invalid intervals events
        if row["is_valid_interval"] != "1":
            continue

        video_id = row["video_id"]
        shot_type = row["shot_type"]

        if video_id in train_videos:
            train_shots[shot_type] += 1

        elif video_id in validation_videos:
            validation_shots[shot_type] += 1

        elif video_id in test_videos:
            test_shots[shot_type] += 1
 

# 5. Print distributions
print("\nTrain shots:", train_shots)
print("\nValidation shots:", validation_shots)
print("\nTest shots:", test_shots)

# 6. save splits
with open(
    "data/metadata/splits.csv",
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(["video_id", "split"])

    for video_id in sorted(train_videos):
        writer.writerow([video_id, "train"])

    for video_id in sorted(validation_videos):
        writer.writerow([video_id, "validation"])

    for video_id in sorted(test_videos):
        writer.writerow([video_id, "test"])