import json
import os
import pandas as pd
from pydub import AudioSegment

def process_audio_segments(json_dir_path, segments_dir, save_dir):

    os.makedirs(segments_dir, exist_ok=True)

    all_segments_info = []

    for json_file_name in os.listdir(json_dir_path):
        if json_file_name.endswith('.json'):
            json_file_path = os.path.join(json_dir_path, json_file_name)

            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            base_name = os.path.splitext(json_file_name)[0]
            wav_file_path = os.path.join(json_dir_path, f'{base_name}.wav')

            if not os.path.exists(wav_file_path):
                print(f"WAV file not found for {json_file_name}. Skipping...")
                continue

            audio = AudioSegment.from_wav(wav_file_path)
            all_segments_info = []

            verbatim_entries = data.get("verbatim", [])

            for i, entry in enumerate(verbatim_entries):
                start_ms = int(entry['start'] * 1000)
                end_ms = int(entry['end'] * 1000)
                text = entry['text']

                segment = audio[start_ms:end_ms]

                segment_file_name = f"{base_name}_segment_{i + 1}.wav"
                segment_file_path = os.path.join(segments_dir, segment_file_name)

                segment.export(segment_file_path, format="wav")

                all_segments_info.append({
                    "filename_segment": text,
                    "start": entry['start'],
                    "end": entry['end'],
                    "path": segment_file_path
                })

    csv_file_path = os.path.join(save_dir, 'all_segments_info.csv')
    df = pd.DataFrame(all_segments_info)
    df.to_csv(csv_file_path, index=False)

    print(f"All segments and metadata saved. CSV file path: {csv_file_path}")
