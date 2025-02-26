from django.core.management.base import BaseCommand
from faker import Faker
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.editor import vfx
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from django.core.files import File

from events.models import Events, AudioFiles, LiveShortVideos
import random
import requests
from PIL import Image
import io
import os


def crossfade(clip1, clip2, transition_duration):
    clip1_end = clip1.duration - transition_duration
    clip2_start = 0

    crossfaded_clip1 = clip1.crossfadeout(transition_duration)
    crossfaded_clip2 = clip2.crossfadein(transition_duration).set_start(clip1_end)

    return CompositeVideoClip([crossfaded_clip1, crossfaded_clip2])


class Command(BaseCommand):
    help = 'Generate short video for a particular event'

    def add_arguments(self, parser):
        parser.add_argument(
            'event_code', type=str,
            help='ID of the event for which to generate a video')

    def handle(self, *args, **kwargs):
        event_code = kwargs['event_code']
        event = Events.objects.get(event_code=event_code)

        # Select up to 5 images randomly related to this event
        images = list(event.event_live_images.all())
        if len(images) < 5:
            selected_images = images
        else:
            selected_images = random.sample(images, 5)

        # Select a random audio file
        audio_files = AudioFiles.objects.first()
        selected_audio = audio_files.audio_file

        # Download the audio file
        audio_file_url = selected_audio.url
        audio_data = requests.get(audio_file_url).content

        # Save the audio data to a local file
        tmp_dir = 'tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        audio_path = f'tmp/{str(selected_audio).split("/")[-1]}'
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_data)

        # Create video clips from images
        clips = []
        report = []  # For reporting
        for image in selected_images:
            # Get the public URL of the image
            image_url = image.image.url

            # Download the image data
            image_data = requests.get(image_url).content

            # Convert the image data to a PIL image
            pil_image = Image.open(io.BytesIO(image_data))

            # Save the PIL image to a local file
            tmp_dir = 'tmp'
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            image_path = f'tmp/{str(image).split("/")[-1]}'
            pil_image.save(image_path)

            # Use the local file to create the ImageSequenceClip
            clip = ImageSequenceClip([image_path], durations=[5])

            # Add the image clip to the final sequence
            clips.append(clip)

            # Delete the local file
            os.remove(image_path)

        # Create a list of crossfaded clips
        crossfaded_clips = []
        for i in range(len(clips) - 1):
            crossfaded_clips.append(crossfade(clips[i], clips[i+1], 1))

        # Concatenate all clips into one video
        final_clip = concatenate_videoclips(crossfaded_clips)

        # Add audio to the video
        audio_clip = AudioFileClip(audio_path)

        # Match the audio duration to the video duration
        if audio_clip.duration > final_clip.duration:
            audio_clip = audio_clip.subclip(0, final_clip.duration)

        # Fade out the audio at the end
        audio_clip = audio_clip.fx(audio_fadeout, 5)

        final_clip = final_clip.set_audio(audio_clip)

        # Save video
        fake = Faker()
        video_filename = f"{fake.unique.file_name(extension='mp4')}"
        video_filepath = f"media/videos/{video_filename}"
        os.makedirs(os.path.dirname(video_filepath), exist_ok=True)
        final_clip.write_videofile(
            video_filepath, fps=24, bitrate="8000k")

        # Save record in the database
        with open(video_filepath, 'rb') as file:
            LiveShortVideos.objects.create(event=event,
                                           file=File(file, name=video_filename))

        # Delete the local video file
        os.remove(video_filepath)
    # Delete the local audio file
    os.remove(audio_path)

    # Print the report
    print('Video Creation Report:')
    print('\n'.join(report))
