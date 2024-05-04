from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx

POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY")
FAL_KEY = os.getenv("FAL_KEY")


class RemoveBackgroundBot(fp.PoeBot):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

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
        message = request.query[-1]
        images = [
            attachment
            for attachment in message.attachments
            if attachment.content_type.startswith("image/")
        ]
        if not images:
            yield fp.PartialResponse(text="Please provide a single image.")
            return
        elif len(images) > 1:
            yield fp.PartialResponse(text="Please provide only one image.")
            return

        yield fp.PartialResponse(text="Removing background...")

        [image] = images
        handle = await self.fal_client.submit(
            "fal-ai/birefnet",
            {"image_url": image.url},
        )

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

        result = await handle.get()
        yield fp.PartialResponse(
            text=f"![image]({result['image']['url']})",
            is_replace_response=True,
        )

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=True,
            introduction_message=(
                "This is a bot that removes the background from the images using BiRefNet model in fal.ai."
            ),
        )


bots = [
    RemoveBackgroundBot(path="/remove-background", access_key=POE_ACCESS_KEY),
]

app = fp.make_app(bots)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
