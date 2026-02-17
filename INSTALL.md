# Installation Guide — Phase Vocoder (Windows 11)

## Step 1: Verify Python is installed

Open **Command Prompt** or **PowerShell** and run:

```
python --version
```

You need **Python 3.10 or newer**. If this fails or shows an older version, download
Python from https://www.python.org/downloads/ and during install **check "Add Python
to PATH"**.

---

## Step 2: Install dependencies

Navigate to the folder containing `requirements.txt` and `phase_vocoder.py`, then run:

```
pip install -r requirements.txt
```

That single command installs everything the app needs.

### If `pip` is not recognised

Try one of these alternatives:

```
python -m pip install -r requirements.txt
```

```
py -m pip install -r requirements.txt
```

### If you get permission errors

Add the `--user` flag:

```
pip install --user -r requirements.txt
```

### If `sounddevice` fails to install

`sounddevice` depends on the PortAudio library. On most Windows machines it works
out of the box, but if you see errors:

1. Make sure you have the **Microsoft Visual C++ Redistributable** installed:
   https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Then retry: `pip install sounddevice`

### If `soundfile` fails to install

`soundfile` needs `libsndfile`. On Windows pip normally bundles it automatically.
If it still fails:

```
pip install soundfile --force-reinstall
```

---

## Step 3: Run the application

```
python phase_vocoder.py
```

---

## Optional: Use a virtual environment (recommended)

This keeps the phase vocoder's packages separate from your system Python:

```
python -m venv vocoder_env
vocoder_env\Scripts\activate
pip install -r requirements.txt
python phase_vocoder.py
```

Each time you want to run the app later, activate the environment first:

```
vocoder_env\Scripts\activate
python phase_vocoder.py
```

---

## Quick reference

| Problem | Fix |
|---|---|
| `python` not found | Use `py` instead, or reinstall Python with "Add to PATH" |
| `pip` not found | Use `python -m pip` instead |
| Permission denied | Add `--user` flag or run as Administrator |
| `sounddevice` error | Install VC++ Redistributable (link above) |
| No audio devices | Check Windows sound settings; make sure a mic/speaker is enabled |
