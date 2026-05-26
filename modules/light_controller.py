"""
modules/light_controller.py

A.T.L.A.S. Bluetooth LED Light Controller

Controls BLE RGB LED bulbs and strips via bleak.
Designed to work with Tuya/Smart Life BLE bulbs (like Vibe E-ssential)
and most generic BLE LED devices.

Supports:
    - on/off
    - color (any CSS color name or hex or rgb)
    - brightness (0-100)
    - effects: flash, fade, strobe, pulse

Author: Sean Doyle / A.T.L.A.S. Project
"""

import asyncio
import logging
from typing import Optional

from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known BLE LED write characteristic UUIDs
# ---------------------------------------------------------------------------

KNOWN_WRITE_UUIDS = [
    "00002b11-0000-1000-8000-00805f9b34fb",  # Tuya BLE
    "0000ffd9-0000-1000-8000-00805f9b34fb",  # Triones / Magic Blue
    "0000ffe9-0000-1000-8000-00805f9b34fb",  # Generic BLE LED
    "00010203-0405-0607-0809-0a0b0c0d2b11",  # Govee / Minger
    "0000fff3-0000-1000-8000-00805f9b34fb",  # ELK-BLEDOM (Amazon strips)
    "0000a002-0000-1000-8000-00805f9b34fb",  # HaoDeng / iLintek
]

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _cmd_triones_color(r, g, b):
    return bytes([0x56, r, g, b, 0x00, 0xf0, 0xaa])

def _cmd_triones_on():
    return bytes([0xcc, 0x23, 0x33])

def _cmd_triones_off():
    return bytes([0xcc, 0x24, 0x33])

def _cmd_triones_brightness(level):
    v = int(level * 255 / 100)
    return bytes([0x56, v, v, v, 0x00, 0xf0, 0xaa])

def _cmd_elk_color(r, g, b):
    return bytes([0x7e, 0x00, 0x05, 0x03, r, g, b, 0x00, 0xef])

def _cmd_elk_on():
    return bytes([0x7e, 0x00, 0x04, 0xf0, 0x00, 0x01, 0xff, 0x00, 0xef])

def _cmd_elk_off():
    return bytes([0x7e, 0x00, 0x04, 0x00, 0x00, 0x00, 0xff, 0x00, 0xef])

def _cmd_elk_brightness(level):
    v = int(level * 255 / 100)
    return bytes([0x7e, 0x00, 0x01, v, 0x00, 0x00, 0x00, 0x00, 0xef])

EFFECT_CODES = {
    "flash":  0x25,
    "strobe": 0x27,
    "fade":   0x26,
    "pulse":  0x28,
}

def _cmd_triones_effect(effect, speed=5):
    code = EFFECT_CODES.get(effect, 0x25)
    return bytes([0xbb, code, speed, 0x44])

# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------

CSS_COLORS = {
    "red":     (255, 0,   0),
    "green":   (0,   255, 0),
    "blue":    (0,   0,   255),
    "white":   (255, 255, 255),
    "warm":    (255, 180, 80),
    "yellow":  (255, 255, 0),
    "orange":  (255, 100, 0),
    "purple":  (128, 0,   128),
    "pink":    (255, 105, 180),
    "cyan":    (0,   255, 255),
    "teal":    (0,   128, 128),
    "magenta": (255, 0,   255),
    "lime":    (0,   255, 0),
    "indigo":  (75,  0,   130),
    "violet":  (238, 130, 238),
    "gold":    (255, 215, 0),
    "silver":  (192, 192, 192),
    "off":     (0,   0,   0),
}

def parse_color(color_str):
    color_str = color_str.strip().lower()

    if color_str in CSS_COLORS:
        return CSS_COLORS[color_str]

    if "warm" in color_str:
        return CSS_COLORS["warm"]

    hex_str = color_str.lstrip("#")
    if len(hex_str) == 6:
        try:
            return (int(hex_str[0:2], 16),
                    int(hex_str[2:4], 16),
                    int(hex_str[4:6], 16))
        except ValueError:
            pass

    if "," in color_str:
        try:
            parts = [int(x.strip()) for x in color_str.split(",")]
            if len(parts) == 3:
                return tuple(max(0, min(255, p)) for p in parts)
        except ValueError:
            pass

    log.warning(f"[Light] Unknown color '{color_str}', defaulting to white")
    return (255, 255, 255)

# ---------------------------------------------------------------------------
# LightController
# ---------------------------------------------------------------------------

class LightController:
    def __init__(self, config=None):
        self.config          = config or {}
        self._client         = None
        self._write_uuid     = None
        self._protocol       = "triones"
        self._device_address = None
        self.debug           = False

        light_cfg = self.config.get("light", {})
        self.enabled         = light_cfg.get("enabled", True)
        self._device_address = light_cfg.get("device_address", None)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    @property
    def is_connected(self):
        return self._client is not None and self._client.is_connected

    async def connect(self, timeout=30.0):
        if not self.enabled:
            return False
        if self.is_connected:
            return True
        try:
            device = await self._find_device(timeout)
            if device is None:
                log.error("[Light] No BLE LED device found")
                return False

            log.info(f"[Light] Connecting to {device.address}...")
            self._client = BleakClient(device, timeout=timeout)
            await self._client.connect()

            if not self._client.is_connected:
                return False

            self._device_address = device.address
            log.info(f"[Light] Connected to {device.address}")
            await self._detect_protocol()
            return True

        except asyncio.TimeoutError:
            log.error("[Light] Connection timed out")
            return False
        except BleakError as e:
            log.error(f"[Light] BLE error: {e}")
            return False
        except Exception as e:
            log.error(f"[Light] Error: {e}")
            return False

    async def _find_device(self, timeout):
        found  = asyncio.Event()
        target = None

        # try saved address first
        if self._device_address:
            def saved_cb(device, adv):
                nonlocal target
                if device.address == self._device_address and target is None:
                    target = device
                    found.set()
            async with BleakScanner(saved_cb):
                try:
                    await asyncio.wait_for(found.wait(), timeout=10)
                    if target:
                        return target
                except asyncio.TimeoutError:
                    log.info("[Light] Saved address not found, scanning...")

        # scan for unnamed BLE device (typical for LED bulbs)
        log.info("[Light] Scanning for BLE LED device...")
        found.clear()
        target = None

        def scan_cb(device, adv):
            nonlocal target
            if target is None and device.name is None:
                target = device
                log.info(f"[Light] Found: {device.address}")
                found.set()

        async with BleakScanner(scan_cb):
            try:
                await asyncio.wait_for(found.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

        return target

    async def _detect_protocol(self):
        if not self._client:
            return

        all_uuids = [
            str(char.uuid)
            for service in self._client.services
            for char in service.characteristics
            if "write" in char.properties
            or "write-without-response" in char.properties
        ]

        log.info(f"[Light] Writable UUIDs: {all_uuids}")

        for known in KNOWN_WRITE_UUIDS:
            if known in all_uuids:
                self._write_uuid = known
                if "ffd9" in known or "ffe9" in known:
                    self._protocol = "triones"
                elif "fff3" in known:
                    self._protocol = "elk"
                elif "2b11" in known:
                    self._protocol = "tuya"
                else:
                    self._protocol = "triones"
                log.info(f"[Light] Protocol: {self._protocol} ({known})")
                return

        if all_uuids:
            self._write_uuid = all_uuids[0]
            self._protocol   = "triones"
            log.info(f"[Light] Fallback UUID: {self._write_uuid}")

    async def disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
        self._client = None

    async def ensure_connected(self):
        if not self.is_connected:
            return await self.connect()
        return True

    async def _send(self, data):
        if not self._write_uuid:
            log.error("[Light] No write characteristic")
            return False
        if not await self.ensure_connected():
            return False
        try:
            await self._client.write_gatt_char(
                self._write_uuid, data, response=False)
            if self.debug:
                log.debug(f"[Light] Sent: {data.hex()}")
            return True
        except BleakError as e:
            log.error(f"[Light] Send failed: {e}")
            self._client = None
            if await self.connect():
                await self._client.write_gatt_char(
                    self._write_uuid, data, response=False)
                return True
            return False

    async def turn_on(self):
        log.info("[Light] ON")
        if self._protocol == "elk":
            return await self._send(_cmd_elk_on())
        return await self._send(_cmd_triones_on())

    async def turn_off(self):
        log.info("[Light] OFF")
        if self._protocol == "elk":
            return await self._send(_cmd_elk_off())
        return await self._send(_cmd_triones_off())

    async def set_color(self, color):
        r, g, b = parse_color(color)
        log.info(f"[Light] Color: {color} → rgb({r},{g},{b})")
        if self._protocol == "elk":
            return await self._send(_cmd_elk_color(r, g, b))
        return await self._send(_cmd_triones_color(r, g, b))

    async def set_brightness(self, level):
        level = max(0, min(100, level))
        log.info(f"[Light] Brightness: {level}%")
        if self._protocol == "elk":
            return await self._send(_cmd_elk_brightness(level))
        return await self._send(_cmd_triones_brightness(level))

    async def set_effect(self, effect, speed=5):
        effect = effect.lower().strip()
        log.info(f"[Light] Effect: {effect} speed={speed}")
        return await self._send(_cmd_triones_effect(effect, speed))

    async def set_rgb(self, r, g, b):
        return await self.set_color(f"{r},{g},{b}")

    async def handle_command(self, command):
        cmd = command.lower().strip()

        if any(w in cmd for w in ["turn on", "switch on", "lights on", "light on"]):
            ok = await self.turn_on()
            return "Lights on." if ok else "Could not reach the light."

        if any(w in cmd for w in ["turn off", "switch off", "lights off", "light off"]):
            ok = await self.turn_off()
            return "Lights off." if ok else "Could not reach the light."

        for effect in EFFECT_CODES:
            if effect in cmd:
                speed = 5
                for word in cmd.split():
                    if word.isdigit():
                        speed = max(1, min(10, int(word)))
                ok = await self.set_effect(effect, speed)
                return f"{effect.capitalize()} mode on." if ok else "Could not reach the light."

        if any(w in cmd for w in ["dim", "brightness", "bright", "percent", "%"]):
            for word in cmd.split():
                word = word.rstrip("%")
                if word.isdigit():
                    ok = await self.set_brightness(int(word))
                    return f"Brightness set to {word} percent." if ok else "Could not reach the light."
            return "Please specify a brightness level, like 'dim to 50'."

        color_triggers = ["set", "change", "color", "colour", "to", "make"]
        if any(t in cmd for t in color_triggers):
            words = cmd.split()
            for i, word in enumerate(words):
                if word.startswith("#") or (len(word) == 6 and all(
                        c in "0123456789abcdef" for c in word)):
                    ok = await self.set_color(word)
                    return f"Color set to {word}." if ok else "Could not reach the light."
                if word in CSS_COLORS:
                    ok = await self.set_color(word)
                    return f"Color set to {word}." if ok else "Could not reach the light."
                if word == "warm" and i + 1 < len(words) and words[i+1] == "white":
                    ok = await self.set_color("warm")
                    return "Warm white on." if ok else "Could not reach the light."

        return "I can control on/off, color, brightness, and effects like flash, strobe, fade, and pulse."


