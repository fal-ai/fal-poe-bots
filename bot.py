from __future__ import annotations

from typing import AsyncIterable

import re
import time
import secrets
import random
import mimetypes
import asyncio
import os
import fal_client
import base64
import fastapi_poe as fp
import httpx
import sentry_sdk

import tempfile
import zipfile

from dataclasses import dataclass, field
from typing import ClassVar

POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY")
SENTRY_DSN = os.getenv("SENTRY_DSN")

if SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )


@dataclass
class BotError(Exception):
    message: str


async def fancy_event_handler(
    handle: fal_client.AsyncRequestHandle,
) -> AsyncIterable[fp.PartialResponse]:
    counter = 0
    async for progress in handle.iter_events(with_logs=True, interval=0.05):
        if isinstance(progress, fal_client.Queued):
            emoji = ["⏳", "⌛"][counter % 2]
            yield fp.PartialResponse(
                text=f"{emoji} Queued... ({progress.position})",
                is_replace_response=True,
            )
        elif isinstance(progress, (fal_client.InProgress, fal_client.Completed)):
            emoji = ["🏃", "🧎"][counter % 2]
            logs = [log["message"] for log in progress.logs][-10:]
            text = f"{emoji} In progress\n```" + "\n".join(logs)
            yield fp.PartialResponse(text=text, is_replace_response=True)

        counter += 1


async def timed_event_handler(
    handle: fal_client.AsyncRequestHandle,
    eta: int | None = None,
    interval: float = 0.05,
) -> AsyncIterable[fp.PartialResponse]:
    start = time.perf_counter()
    async for progress in handle.iter_events(with_logs=True, interval=interval):
        timing = int(time.perf_counter() - start)
        if eta is None or timing > eta:
            text = f"Generating image ({timing}s elapsed)"
        else:
            text = f"Generating image ({int(time.perf_counter() - start)}/{eta}s elapsed)"

        yield fp.PartialResponse(
            text=text,
            is_replace_response=True,
        )


async def response_with_data_url(
    bot: FalBaseBot, request: fp.QueryRequest, url: str
) -> fp.PartialResponse:
    if not url.startswith("data:"):
        return fp.PartialResponse(
            text=f"![image]({url})",
            is_replace_response=True,
        )

    # Parse a base64 encoded image data URL and return a response with the image.
    content_type, raw_data = url.removeprefix("data:").split(";", 1)
    encoding, data = raw_data.split(",", 1)
    assert encoding == "base64"
    image_data = base64.b64decode(data)
    extension = mimetypes.guess_extension(content_type) or ".jpeg"

    await bot.post_message_attachment(
        message_id=request.message_id,
        file_data=image_data,
        filename="image" + extension,
    )
    return fp.PartialResponse(
        text="The image is too large to display here, but it has been sent as an attachment."
    )


def parse_image(request: fp.QueryRequest) -> fp.Attachment:
    images = [
        attachment
        for attachment in request.query[-1].attachments
        if attachment.content_type.startswith("image/")
    ]
    if not images:
        raise BotError(
            "No images found, please provide a single image as an attachment."
        )
    elif len(images) > 1:
        raise BotError("More than one images are found, please provide only one image.")
    return images[0]


def parse_images(request: fp.QueryRequest) -> list[fp.Attachment]:
    images = [
        attachment
        for attachment in request.query[-1].attachments
        if attachment.content_type.startswith("image/")
    ]
    if not images:
        raise BotError(
            "No images found, please provide a single image as an attachment."
        )
    return images


@dataclass
class FalBaseBot(fp.PoeBot):
    INTRO_MESSAGE: ClassVar[str | None] = "This is a demo bot powered by fal.ai."

    fal_client: fal_client.AsyncClient = field(default_factory=fal_client.AsyncClient)
    http_client: httpx.AsyncClient = field(default_factory=httpx.AsyncClient)

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        yield fp.MetaResponse(
            text="",
            content_type="text/markdown",
            linkify=True,
            refetch_settings=False,
            suggested_replies=False,
        )
        try:
            async for response in self.execute(request):
                yield response
        except BotError as e:
            yield fp.PartialResponse(text=str(e))

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        raise NotImplementedError
        yield

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        kwargs = {}
        if self.INTRO_MESSAGE:
            kwargs["intro_message"] = self.INTRO_MESSAGE
        return fp.SettingsResponse(
            allow_attachments=True,
            enable_multi_bot_chat_prompting=True,
            **kwargs,
        )


class RemoveBackgroundBot(FalBaseBot):
    INTRO_MESSAGE = "This is a bot that removes the background from the images using BiRefNet model in fal.ai."

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        image = parse_image(request)

        yield fp.PartialResponse(text="Removing background...")

        handle = await self.fal_client.submit(
            "fal-ai/birefnet",
            {"image_url": image.url},
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        yield fp.PartialResponse(
            text=f"![image]({result['image']['url']})",
            is_replace_response=True,
        )


class CreativeUpscale(FalBaseBot):
    INTRO_MESSAGE = "This is a bot that upscales the images using the given prompt. Not intended for serious use."

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            prompt = "ultra hd, 4k, high quality"
            yield fp.PartialResponse(
                text=f"No prompt provided with the image, using the default prompt: {prompt!r}\n"
            )
            await asyncio.sleep(0.2)

        image = parse_image(request)
        yield fp.PartialResponse(text="Upscaling the image...")
        handle = await self.fal_client.submit(
            "fal-ai/controlnet-tile-upscaler",
            {"image_url": image.url, "prompt": prompt},
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        yield fp.PartialResponse(
            text=f"![image]({result['image']['url']})",
            is_replace_response=True,
        )


class AnimagineXL(FalBaseBot):
    INTRO_MESSAGE = (
        "This is a bot that can generate anime-themed pictures from your prompts. Prompt format: '1girl/1boy, character name, "
        "from what series, everything else in any order.'"
    )

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        yield fp.PartialResponse(text="Generating anime-themed picture...")
        handle = await self.fal_client.submit(
            "fal-ai/any-sd",
            arguments={
                "model_name": "cagliostrolab/animagine-xl-3.1",
                "image_size": {
                    "width": 832,
                    "height": 1216,
                },
                "prompt": "masterpiece, best quality, very aesthetic, absurdres, "
                + prompt,
                "negative_prompt": "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
                "guidance_scale": 7,
                "num_inference_steps": 28,
                "safety_checker_version": "v2",
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class RealVisXL(FalBaseBot):
    INTRO_MESSAGE = (
        "This is a bot that can generate realistic pictures from your prompts."
    )

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        yield fp.PartialResponse(text="Generating a realistic picture...")
        handle = await self.fal_client.submit(
            "fal-ai/realistic-vision",
            arguments={
                "model_name": "SG161222/RealVisXL_V4.0",
                "image_size": {
                    "width": 832,
                    "height": 1216,
                },
                "prompt": prompt
                + "8k resolution, best quality, beautiful photograph, dynamic lighting",
                "negative_prompt": "NSFW, nudity, worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, sexual, nude, nudity, human anatomy",
                "guidance_scale": 7,
                "num_inference_steps": 25,
                "safety_checker_version": "v2",
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class PixelArtBot(FalBaseBot):
    INTRO_MESSAGE = (
        "This is a bot that can generate pixel art styled pictures from your prompts."
    )

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        yield fp.PartialResponse(text="Generating a pixel art for you...")
        handle = await self.fal_client.submit(
            "fal-ai/fast-sdxl",
            arguments={
                "prompt": "pixel, " + prompt,
                "negative_prompt": "NSFW, nudity, overexposed, sexual, nude, nudity, human anatomy, 3D render, cinematic",
                "guidance_scale": 7,
                "num_inference_steps": 25,
                "loras": [
                    {
                        "path": "https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors?download=true",
                        "scale": 1.0,
                    }
                ],
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class TurboTextToVideoBot(FalBaseBot):
    INTRO_MESSAGE = "This is a bot that can generate videos from your prompts. Try with stuff like 'a dog wearing vr goggles on a boat'."

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        yield fp.PartialResponse(text="Generating the video for you...")
        handle = await self.fal_client.submit(
            "fal-ai/t2v-turbo",
            arguments={
                "prompt": prompt,
                "num_inference_steps": random.choice([4, 8]),
                "num_frames": 32,
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        await self.post_message_attachment(
            message_id=request.message_id,
            download_url=result["video"]["url"],
        )
        yield fp.PartialResponse(text="Video created!", is_replace_response=True)


class StableDiffusionv32B(FalBaseBot):
    INTRO_MESSAGE = "Generate images with stability's latest model with your prompts. Powered by fal.ai."

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        yield fp.PartialResponse(text="Generating a picture...")
        handle = await self.fal_client.submit(
            "fal-ai/stable-diffusion-v3-medium",
            arguments={
                "prompt": prompt,
                "negative_prompt": "NSFW, nudity, worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, sexual, nude, nudity, human anatomy",
                "guidance_scale": 6,
                "num_inference_steps": 28,
                "safety_checker_version": "v2",
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class LivePortrait(FalBaseBot):
    INTRO_MESSAGE = (
        "Animates given portraits with the motion's in the video. Powered by fal.ai."
    )

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        attachments = request.query[-1].attachments
        if not len(attachments) == 2:
            raise BotError("Please provide a video and a portrait image.")

        videos = [
            attachment
            for attachment in attachments
            if attachment.content_type.startswith("video/")
        ]
        if not videos:
            raise BotError("No video found, please provide a video as an attachment.")
        elif len(videos) > 1:
            raise BotError(
                "More than one video is found, please provide only one video."
            )
        [video] = videos

        images = [
            attachment
            for attachment in attachments
            if attachment.content_type.startswith("image/")
        ]
        if not images:
            raise BotError(
                "No image found, please provide a single image as an attachment."
            )
        elif len(images) > 1:
            raise BotError(
                "More than one images are found, please provide only one image."
            )
        [image] = images

        yield fp.PartialResponse(text="Animating the portrait...")
        handle = await self.fal_client.submit(
            "fal-ai/live-portrait",
            {
                "video_url": video.url,
                "image_url": image.url,
            },
        )

        async for event in fancy_event_handler(handle):
            yield event

        try:
            result = await handle.get()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise BotError(e.response.json()["detail"][-1]["msg"])
            raise

        response = await self.http_client.get(
            result["video"]["url"], follow_redirects=True
        )
        response.raise_for_status()

        await self.post_message_attachment(
            message_id=request.message_id,
            file_data=response.content,
            filename="video.mp4",
            content_type=response.headers["Content-Type"],
        )
        yield fp.PartialResponse(text=" ", is_replace_response=True)


OPTION_RE = re.compile(r"--(\w+)\s+([^\s]+)")


def extract_options(input_string):
    matches = OPTION_RE.findall(input_string)
    options = {key: value for key, value in matches}
    cleaned_string = OPTION_RE.sub("", input_string)
    cleaned_string = " ".join(cleaned_string.split())
    return options, cleaned_string


def parse_aspect_ratio(aspect_ratio, max_size=1024):
    width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
    scale = min(max_size / width_ratio, max_size / height_ratio)
    width = int(width_ratio * scale)
    height = int(height_ratio * scale)
    return width, height


@dataclass
class ParsedPrompt:
    prompt: str
    width: int
    height: int
    options: dict[str, str]

    @classmethod
    def from_raw(cls, raw_prompt: str) -> ParsedPrompt:
        options, prompt = extract_options(raw_prompt)
        try:
            width, height = parse_aspect_ratio(options.get("aspect", "4:3"))
        except (ValueError, ArithmeticError):
            raise BotError(
                "Invalid aspect ratio provided, use the format 'width_scale:height_scale', like '1:1' or '4:3'."
            )
        return cls(prompt=prompt, width=width, height=height, options=options)


class FluxPro(FalBaseBot):
    INTRO_MESSAGE = None

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        parsed_prompt = ParsedPrompt.from_raw(prompt)
        handle = await self.fal_client.submit(
            "fal-ai/flux-pro",
            arguments={
                "prompt": parsed_prompt.prompt,
                "image_size": {
                    "width": parsed_prompt.width,
                    "height": parsed_prompt.height,
                },
                "safety_tolerance": "5",
                "num_inference_steps": 40,
            },
        )

        async for event in timed_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class FluxDev(FalBaseBot):
    INTRO_MESSAGE = None

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        parsed_prompt = ParsedPrompt.from_raw(prompt)
        handle = await self.fal_client.submit(
            "fal-ai/flux/dev",
            arguments={
                "prompt": parsed_prompt.prompt,
                "image_size": {
                    "width": parsed_prompt.width,
                    "height": parsed_prompt.height,
                },
                "num_inference_steps": 40,
            },
        )

        async for event in timed_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


class FluxSchnell(FalBaseBot):
    INTRO_MESSAGE = None

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        parsed_prompt = ParsedPrompt.from_raw(prompt)
        handle = await self.fal_client.submit(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": parsed_prompt.prompt,
                "image_size": {
                    "width": parsed_prompt.width,
                    "height": parsed_prompt.height,
                },
            },
        )

        async for event in timed_event_handler(handle):
            yield event

        result = await handle.get()
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


async def download_and_zip_images(
    client: httpx.AsyncClient,
    images: list[fp.Attachment],
) -> str:
    """
    Download all the image urls in parallel to a temporary directory,
    zip them, and upload to fal with fal_client.upload_file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download images
        download_tasks = [download_image(client, image, temp_dir) for image in images]
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Process results
        downloaded_files = []
        failed_downloads = 0
        for result in download_results:
            if result:
                downloaded_files.append(result)
            else:
                failed_downloads += 1

        # Check if too many downloads failed
        if failed_downloads > len(images) // 2:
            raise Exception(
                f"More than 50% of images failed to download ({failed_downloads}/{len(images)})"
            )

        if not downloaded_files:
            raise Exception("No images were successfully downloaded")

        # Create zip file
        zip_path = os.path.join(temp_dir, "images.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for file_path in downloaded_files:
                zip_file.write(file_path, os.path.basename(file_path))

        uploaded_file_id = await fal_client.upload_file_async(zip_path)
        return uploaded_file_id


async def download_image(
    client: httpx.AsyncClient, image: fp.Attachment, temp_dir: str
) -> str | None:
    """Download a single image and return its file path."""
    if not (extension := mimetypes.guess_extension(image.content_type)):
        print(f"Unknown content type: {image.content_type}")
        return None

    file_name = secrets.token_hex(8) + extension
    try:
        response = await client.get(image.url)
        response.raise_for_status()
        file_name = os.path.join(temp_dir, file_name)
        with open(file_name, "wb") as f:
            f.write(response.content)
        return file_name
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred while downloading {image.url}: {e}")
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {image.url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {image.url}: {e}")
    return None


class FluxFinetuning(FalBaseBot):
    INTRO_MESSAGE = None

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        images = parse_images(request)
        if len(images) <= 4:
            raise BotError("Please provide at least 5 images for fine-tuning.")
        elif len(images) > 20:
            raise BotError("Please provide less than 20 images for fine-tuning.")

        zip_url = await download_and_zip_images(
            self.http_client,
            images=images,
        )
        handle = await self.fal_client.submit(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": zip_url,
                "trigger_word": "ohwx",
                "iter_multiplier": min(1, len(images) * 0.1),
            },
        )

        async for event in timed_event_handler(handle, interval=0.8, eta=300):
            yield event

        result = await handle.get()
        yield fp.PartialResponse(
            text=f"key: `{result['diffusers_lora_file']['url']}`",
            is_replace_response=True,
        )


class FluxDevFinetunes(FalBaseBot):
    INTRO_MESSAGE = None

    async def execute(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        prompt = request.query[-1].content
        if not prompt:
            raise BotError("No prompt provided with the image.")

        parsed_prompt = ParsedPrompt.from_raw(prompt)
        if not (lora := parsed_prompt.options.get("lora")):
            raise BotError("Please provide a lora key with the prompt.")

        handle = await self.fal_client.submit(
            "fal-ai/flux-lora",
            arguments={
                "prompt": parsed_prompt.prompt,
                "image_size": {
                    "width": parsed_prompt.width,
                    "height": parsed_prompt.height,
                },
                "num_inference_steps": 40,
                "loras": [
                    {
                        "path": lora,
                        "scale": 1.0,
                    }
                ],
            },
        )

        async for event in timed_event_handler(handle):
            yield event

        result = await handle.get()
        print(result)
        if result["has_nsfw_concepts"][0]:
            yield fp.PartialResponse(
                text="The generated image contains NSFW content, please try again with a different prompt.",
                is_replace_response=True,
            )
            return

        yield (await response_with_data_url(self, request, result["images"][0]["url"]))


bots = [
    RemoveBackgroundBot(path="/remove-background", access_key=POE_ACCESS_KEY),
    CreativeUpscale(path="/creative-upscaler", access_key=POE_ACCESS_KEY),
    AnimagineXL(path="/animagine-xl", access_key=POE_ACCESS_KEY),
    RealVisXL(path="/real-vis-xl", access_key=POE_ACCESS_KEY),
    PixelArtBot(path="/pixel-art", access_key=POE_ACCESS_KEY),
    TurboTextToVideoBot(path="/turbo-text-to-video", access_key=POE_ACCESS_KEY),
    StableDiffusionv32B(path="/stable-diffusion-v3-2b", access_key=POE_ACCESS_KEY),
    LivePortrait(path="/live-portrait", access_key=POE_ACCESS_KEY),
    FluxPro(path="/flux-pro", access_key=POE_ACCESS_KEY),
    FluxDev(path="/flux-dev", access_key=POE_ACCESS_KEY),
    FluxSchnell(path="/flux-schnell", access_key=POE_ACCESS_KEY),
    FluxFinetuning(path="/flux-finetuning", access_key=POE_ACCESS_KEY),
    FluxDevFinetunes(path="/flux-dev-finetunes", access_key=POE_ACCESS_KEY),
]

app = fp.make_app(bots)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
