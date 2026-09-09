# Running this project on Windows

Tested against the same code that runs on macOS. Nothing platform-specific is
needed beyond the commands below - paths use `pathlib`, and the device
selection falls back automatically from Metal to CUDA to CPU.

## What you actually need

| Goal | Dataset (8 GB) | Trained weights (60 MB) |
|------|----------------|--------------------------|
| Run the demo app on your own images | no | **yes** |
| Reproduce the reported test metrics | yes | **yes** |
| Retrain from scratch | **yes** | no |
| Regenerate dataset/EDA figures | **yes** | no |

Most people only want the first row, which needs the weights and nothing else.

---

## 1. Install Python

Download Python **3.12 or newer** from [python.org/downloads](https://www.python.org/downloads/).

During the installer, tick **"Add python.exe to PATH"**. This is the step
people miss, and every later command fails without it.

Verify in a new PowerShell window:

```powershell
py --version
```

## 2. Get the code

**Option A - download the ZIP (no Git needed, recommended)**

1. Open the
   [repository page](https://github.com/ns8963038-hub/Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy)
2. Green **Code** button -> **Download ZIP**
3. Extract it to your Desktop. The extracted folder keeps a `-main` suffix.
4. Point the terminal at it:

```powershell
cd $env:USERPROFILE\Desktop\Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy-main
```

**Option B - clone with Git**

Only if Git is installed. Check with `git --version`; if that errors, use
Option A or install Git from [git-scm.com](https://git-scm.com/download/win)
and **open a new terminal** afterwards, because PATH changes only apply to new
windows.

```powershell
git clone https://github.com/ns8963038-hub/Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy.git
cd Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy
```

### Check you are in the right folder before going further

```powershell
dir
```

You should see `README.md`, `requirements.txt`, `src` and `app` listed. **If
you do not, stop here** - every command below will fail, and they fail in ways
that point at the wrong problem.

> This is the single most common way the setup goes wrong. If `git clone`
> fails because Git is not installed, no folder is created, the following `cd`
> also fails, and the terminal stays in your home directory. The commands after
> that then report *"Could not open requirements file"*, *"No module named
> 'src'"* and *"streamlit is not recognized"* - three unrelated-looking errors
> that all trace back to the clone never having happened.

## 3. Create the virtual environment

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`.

> If PowerShell refuses with *"running scripts is disabled on this system"*,
> run this once, then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> Or use Command Prompt instead of PowerShell, where the activate command is
> `.venv\Scripts\activate.bat`.

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls PyTorch built for **CPU**, which is all you need to run the demo.

**If the laptop has an NVIDIA GPU** and you intend to retrain, install the
CUDA build instead (check your driver version first with `nvidia-smi`):

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Confirm what PyTorch found:

```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## 5. Download the trained weights

```powershell
python -m src.download_weights
```

This pulls two checkpoints (~60 MB total) into `outputs\models\`. They are
published as GitHub release assets rather than committed, so clones stay
small.

Manual alternative: download them from the
[v1.0 release](https://github.com/ns8963038-hub/Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy/releases/tag/v1.0)
and drop both `.pt` files into `outputs\models\`.

## 6. Run the demo

```powershell
streamlit run app/app.py
```

A browser tab opens at `http://localhost:8501`. Upload any retinal fundus
photograph and you get the predicted grade, per-class confidence, the
referable-DR decision, and a Grad-CAM overlay.

**That is everything for a demo.** The steps below are only for retraining or
regenerating figures.

---

## Optional: the dataset

Only needed to retrain, to evaluate on the test split, or to regenerate the
dataset figures. It is roughly 8 GB.

```powershell
pip install kaggle
kaggle auth login
python -m src.download_data
python -m src.preprocess
```

`kaggle auth login` opens a browser once and caches the credentials.

## Optional: retraining

There is no `make` on Windows, so call the modules directly:

```powershell
python -m src.train --model baseline --workers 2
python -m src.train --model efficientnet --workers 2
python -m src.evaluate --model efficientnet --split test
python -m src.compare --split test
```

Two Windows-specific notes:

- **Use `--workers 2` (or `0`).** Windows spawns worker processes rather than
  forking, so each worker re-imports PyTorch. Four workers costs more in
  startup than it wins in throughput on most laptops.
- **Expect this to be slow on CPU.** EfficientNet-B3 takes roughly 2 minutes
  per epoch on an Apple M5 GPU; on a CPU-only laptop budget 15-30 minutes per
  epoch. Use the released weights unless you specifically need to retrain.

Quick end-to-end check without committing to a full run:

```powershell
python -m src.train --model efficientnet --epochs 2 --limit 64 --workers 0
```

---

## Troubleshooting

**`'python' is not recognized`** - Python was installed without "Add to PATH".
Re-run the installer, choose Modify, and enable it. Or use `py` instead.

**`ModuleNotFoundError: No module named 'src'`** - You are not in the project
root. Run `dir`: if it does not list `README.md` and `src`, `cd` to the folder
that does. Run modules as `python -m src.<name>`, not `python src\<name>.py`.

**`'git' is not recognized`** - Git is not installed. Use the ZIP download in
step 2 instead; nothing in this project needs Git to run.

**`Could not open requirements file`** - Same cause as above: the terminal is
not in the project folder. Check with `dir`.

**`No trained model found` in the app** - Step 5 has not run, or the files
landed elsewhere. Check that `outputs\models\efficientnet_best.pt` exists.

**Streamlit opens but the page is blank** - Some corporate or antivirus
software blocks the local websocket. Try
`streamlit run app/app.py --server.port 8080`, or a different browser.

**`OSError: [WinError 1455] The paging file is too small`** - Too many
DataLoader workers for the available RAM. Add `--workers 0`.

**Slow first prediction** - The first inference loads the model and warms up
PyTorch. Subsequent predictions are fast; the model stays cached.
