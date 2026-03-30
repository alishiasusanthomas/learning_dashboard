from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_path, output_audio_path):
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is None:
            raise ValueError("The uploaded video does not contain an audio track.")
        audio.write_audiofile(output_audio_path)
        audio.close()
