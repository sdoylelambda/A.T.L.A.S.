import logging
import asyncio

from modules.light_controller import LightController


async def _test():
    logging.basicConfig(level=logging.INFO)
    light = LightController()
    print("Connecting...")
    if not await light.connect():
        print("Could not connect.")
        return
    print("ON");     await light.turn_on();           await asyncio.sleep(1)
    print("RED");    await light.set_color("red");     await asyncio.sleep(1)
    print("GREEN");  await light.set_color("green");   await asyncio.sleep(1)
    print("BLUE");   await light.set_color("blue");    await asyncio.sleep(1)
    print("WHITE");  await light.set_color("white");   await asyncio.sleep(1)
    print("FLASH");  await light.set_effect("flash");  await asyncio.sleep(3)
    print("FADE");   await light.set_effect("fade");   await asyncio.sleep(3)
    print("OFF");    await light.turn_off()
    await light.disconnect()
    print("Done!")
