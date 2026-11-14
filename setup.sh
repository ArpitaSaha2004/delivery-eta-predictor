#!/bin/bash
# Run this once from inside the delivery-eta-predictor folder:  bash setup.sh
set -e

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing libraries (this can take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name delivery-eta --display-name "Python (delivery-eta)"

echo ""
echo "Setup finished."
echo "Next: put train.csv inside the data/ folder, then open notebooks/step1.ipynb in VS Code"
echo "and choose the kernel 'Python (delivery-eta)' or the one inside venv."
echo "Next time, activate the environment with:  source venv/bin/activate"
