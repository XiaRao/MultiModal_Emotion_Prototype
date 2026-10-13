import json
import time
import pandas as pd
from pathlib import Path
from inference import predict_emotion

dev_df = pd.read_csv("data/annotations/dev_sent_emo.csv")
subset = dev_df.head(10)

stream_started = time.perf_counter()
audio_dir = Path("data/audio/dev")

for index, row in subset.iterrows():
    text = row["Utterance"]
    dia_id = int(row["Dialogue_ID"])
    utt_id = int(row["Utterance_ID"])
    audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.wav"

    if not audio_path.exists():
        print(f"Missing audio: {audio_path.name}")
        continue

    started = time.perf_counter()
    result = predict_emotion(text, audio_path)
    latency_ms = (time.perf_counter() - started) * 1000

    event = {
        "dialogue_id": dia_id,
        "utterance_id": utt_id,
        "speaker": str(row["Speaker"]),
        "text": text,
        "received_at_s": round(started - stream_started, 3),
        "emotion": result["emotion"],
        "response": result["response"],
        "processing_ms": round(latency_ms, 2)
    }

    print(json.dumps(event, ensure_ascii=False), flush=True)
    time.sleep(1)
