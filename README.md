# CourtVision

## 1. Problem Definition

### Objective

CourtVision analyzes short basketball video clips and automatically identifies **whether a shot occurs, when the shooting event occurs, and what type of shot was attempted**.

### Input

A short basketball video clip containing either a shooting sequence or non-shooting activity.

Example:

```text
basketball_clip.mp4
```

### Output

For each input video, the system produces:

```text
Shot detected: YES
Shot type: Jump Shot
Shot timestamp: 2.14 s
Confidence: 0.91
```

The production API will eventually return the result in structured form:

```json
{
  "shot_detected": true,
  "shot_type": "jump_shot",
  "shot_timestamp": 2.14,
  "confidence": 0.91
}
```

---

### ML Tasks

CourtVision consists of two primary ML tasks.

#### Task 1 — Temporal Shot Detection

Determine **whether and where a shooting event occurs within a video sequence**.

```text
Video frames
     ↓
Temporal model
     ↓
Shot probability over time
     ↓
Detected shot timestamp
```

The model should distinguish between:

* `SHOT`
* `NO_SHOT`

and identify the approximate temporal location of the shooting event.

#### Task 2 — Shot Type Classification

Given a detected shooting sequence, classify the type of shot.

Initial target classes:

* `JUMP_SHOT`
* `LAYUP`
* `FREE_THROW`
* `OTHER`

The final class taxonomy will be confirmed after inspecting the available dataset and class distribution.

---

## Scope

### Included

* Basketball video preprocessing
* Temporal shot-event detection
* Shot-type classification
* Model training and evaluation
* Reproducible data preprocessing
* Experiment tracking
* Model versioning
* Model registry
* API-based inference
* Dockerized model serving
* Automated testing
* CI/CD
* Deployment
* Basic inference monitoring

### Not Included

The following are intentionally outside the initial scope:

* Player detection
* Ball detection
* Multi-object tracking
* Pose estimation
* 3D biomechanics
* Release-angle estimation
* Make/miss prediction
* Full-game analytics
* Custom object detection
* Large-scale distributed training

These may be considered future extensions but are **not part of the initial project goal**.

---

## Success Criteria

The project will be considered complete when it can:

1. Accept a basketball video clip as input.
2. Determine whether a shot occurs.
3. Identify the approximate shot timestamp.
4. Classify the shot type.
5. Return the result through an inference API.
6. Load a versioned model from the model registry.
7. Run inference inside a Docker container.
8. Automatically run tests through CI/CD.
9. Track model experiments and metrics with MLflow.
10. Record basic inference information such as model version, confidence, and latency.
