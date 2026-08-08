import os
import pandas as pd
import numpy as np
import torchaudio
import yt_dlp
import concurrent.futures
import time

csv_fp = input('Enter the csv filepath: ')
output_fp = input('Enter the download directory: ')

videos = pd.read_csv(csv_fp)
audio_arrays = []
no_download = []
failed_ids = []

def download_audio(video_id, ydl_opts):
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=' + video_id])
        return True
    except Exception as e:
        print(f'https://www.youtube.com/watch?v={video_id} failed: {e}')
        return False

for index, row in videos.iterrows():
    video_id = row['YouTube ID']
    start_time = row['start seconds'] // 1000
    end_time = row['end seconds'] // 1000

    ydl_opts = {
        'verbose': False,
        'quiet': True,
        'no_warnings': True,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': os.path.join(output_fp, f'{video_id}.%(ext)s')
    }

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(download_audio, video_id, ydl_opts)
            try:
                result = future.result(timeout=20)
            except concurrent.futures.TimeoutError:
                print(f'https://www.youtube.com/watch?v={video_id} timed out after 20 seconds')
                result = False

        if result:
            if os.path.exists(os.path.join(output_fp, video_id) + '.m4a'):
                os.remove(os.path.join(output_fp, video_id) + '.m4a')

            file_path = os.path.join(output_fp, video_id) + '.wav'
            waveform, sample_rate = torchaudio.load(file_path)

            try:
                trimmed_waveform = waveform[:, sample_rate * start_time:sample_rate * end_time]
                audio_arrays.append((trimmed_waveform, sample_rate))
                torchaudio.save(file_path, trimmed_waveform, sample_rate)

            except Exception as e:
                print(f'https://www.youtube.com/watch?v={video_id} failed time processing: {e}')
                audio_arrays.append(np.nan)
                no_download.append(video_id)
        else:
            audio_arrays.append(np.nan)
            no_download.append(video_id)

    except Exception as e:
        print(f'https://www.youtube.com/watch?v={video_id} failed: {e}')
        if os.path.exists(os.path.join(output_fp, video_id) + '.wav'):
            os.remove(os.path.join(output_fp, video_id) + '.wav')
        if os.path.exists(os.path.join(output_fp, video_id) + '.m4a'):
            os.remove(os.path.join(output_fp, video_id) + '.m4a')

        audio_arrays.append(np.nan)
        no_download.append(video_id)

print(f'{len(no_download)} videos unavailable')
videos['audio'] = audio_arrays
output_csv = csv_fp[:-4] + 'Audio.csv'
videos.to_csv(output_csv)
print(f"Output csv downloaded as {output_csv} with audio tensors")
