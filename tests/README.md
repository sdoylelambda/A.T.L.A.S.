# ATLAS Tests

## Install dependencies
```bash
pip install pytest pytest-asyncio
```

## Run all tests
```bash
pytest tests/ -v
```

## Run a specific file
```bash
pytest tests/test_ears.py -v
pytest tests/test_eyes.py -v
pytest tests/test_stt.py -v
pytest tests/test_launcher.py -v
pytest tests/test_calendar.py -v
```

## Structure
| File               | Covers                                                                           |
|--------------------|----------------------------------------------------------------------------------|
| `test_ears.py`     | RMS calculation, noise floor calibration, speech detection                       |
| `test_eyes.py`     | item detection, image description, color identification, verifies hardware       |
| `test_stt.py`      | float32 conversion, short/long routing, minimum duration guard                   |
| `test_launcher.py` | App launching, command routing, browser page alive check, false-match prevention |
| `test_calendar.py` | Check events for today, tomorrow, this week, and add events to Google calendar   |
