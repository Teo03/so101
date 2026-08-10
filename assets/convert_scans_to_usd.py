"""Convert the scanned GLBs to textured USD using Isaac Sim's converter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = ROOT / "assets" / "scans"


async def convert(input_path: Path, output_path: Path) -> None:
    import omni.kit.asset_converter as converter

    context = converter.AssetConverterContext()
    context.embed_textures = True
    context.export_preview_surface = True
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.use_meter_as_world_unit = True
    task = converter.get_instance().create_converter_task(
        str(input_path), str(output_path), None, context
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"Failed to convert {input_path}: {task.get_error_message()}")
    print(f"[convert] wrote {output_path}", flush=True)


def main() -> None:
    app = SimulationApp({"headless": True, "renderer": "None"})
    try:
        asyncio.get_event_loop().run_until_complete(
            convert(SCAN_DIR / "basket.glb", SCAN_DIR / "basket_textured.usd")
        )
        asyncio.get_event_loop().run_until_complete(
            convert(SCAN_DIR / "chocolate_bar.glb", SCAN_DIR / "chocolate_bar_textured.usd")
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
