#!/bin/bash
# Double-click to launch the demo on macOS.
cd "$(dirname "$0")"
if [ ! -f ".venv/bin/activate" ]; then
  echo "No virtual environment found. Run the setup in README.md first."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi
source .venv/bin/activate
[ -f "outputs/models/efficientnet_best.pt" ] || python -m src.download_weights
echo "Starting the app. Press Ctrl+C to stop."
streamlit run app/app.py
