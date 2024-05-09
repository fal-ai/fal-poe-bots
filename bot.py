from __future__ import annotations

from typing import AsyncIterable

import mimetypes
import asyncio
import os
import fal_client
import base64
import fastapi_poe as fp
import httpx
import sentry_sdk

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
        file_name="image" + extension,
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


@dataclass
class FalBaseBot(fp.PoeBot):
    INTRO_MESSAGE: ClassVar[str] = "This is a demo bot powered by fal.ai."

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
        return fp.SettingsResponse(
            allow_attachments=True,
            introduction_message=self.INTRO_MESSAGE,
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


bots = [
    RemoveBackgroundBot(path="/remove-background", access_key=POE_ACCESS_KEY),
    CreativeUpscale(path="/creative-upscaler", access_key=POE_ACCESS_KEY),
    AnimagineXL(path="/animagine-xl", access_key=POE_ACCESS_KEY),
    RealVisXL(path="/real-vis-xl", access_key=POE_ACCESS_KEY),
    PixelArtBot(path="/pixel-art", access_key=POE_ACCESS_KEY),
]

app = fp.make_app(bots)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
