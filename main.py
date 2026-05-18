import yaml
import threading
import sys
import asyncio
import os
import argparse

from modules.window_controller import WindowController


def main():
    brain = Brain()
    hands = Hands()
    awareness = Awareness()
    print("Jarvis (single-node) running...")

    while True:
        intent = awareness.listen()

        if not intent:
            time.sleep(0.5)
            continue

        plan = brain.create_plan(intent)
        awareness.speak(plan['summary'])
        approval = awareness.listen_confirmation()

        if approval:
            awareness.speak(plan['summary'])
            hands.execute(plan)
        else:
            awareness.speak("Plan cancelled.")


if __name__ == "__main__":
    main()
