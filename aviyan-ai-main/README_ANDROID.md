# AVIYAN Android APK

AVIYAN's Android client is a lightweight Python/Kivy application. The Android package intentionally does **not** bundle the large 1B/5B training stack or model weights. The phone app connects to the AVIYAN backend so the APK stays buildable and practical.

## Local test

```bash
cd mobile_app
python -m pip install -r requirements.txt
python main.py
```

Set a backend URL with:

```text
AVIYAN_API_URL=http://YOUR_SERVER:8000
```

Android emulator uses `http://10.0.2.2:8000` by default.

## Build APK on GitHub

1. Push this repository to GitHub.
2. Open **Actions**.
3. Select **Build AVIYAN Android APK**.
4. Click **Run workflow**.
5. Wait for the workflow to finish.
6. Open the workflow run and download the `AVIYAN-debug-apk` artifact.

The workflow uses Buildozer/python-for-android and produces an ARM Android debug APK.

## Backend

The existing FastAPI backend remains in `api/main.py`. Run it from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Important capability boundary

The app UI exposes chat, reasoning, coding, image and video entry points. Real image/video generation requires a compatible generation backend/model. The project does not pretend that a UI alone is a trained image/video model. Providers can be connected later without changing the Android shell.

## Identity

AVIYAN was specified as created/developed by **ABISHEK BHUSAL**, Founder & Creator of **AB DEV STUDIO**, from **Nepal**. The app UI is English-first and the assistant should reply in the user's language.
