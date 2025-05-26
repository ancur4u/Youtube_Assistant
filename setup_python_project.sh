#!/bin/bash

# Self-chmod if not already executable (future-proof)
if [ ! -x "$0" ]; then
  echo "🔐 Making script executable for future use..."
  chmod +x "$0"
fi

# === Step 1: Detect the latest Python version installed ===
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then
  echo "❌ Python is not installed."
  exit 1
fi

echo "✅ Using Python: $PYTHON"

# === Step 2: Create virtual environment ===
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  $PYTHON -m venv venv
else
  echo "✔️  venv already exists. Skipping creation."
fi

# === Step 3: Activate the virtual environment ===
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# === Step 4: Upgrade pip ===
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# === Step 5: Install from requirements.txt if it exists ===
if [ -f "requirements.txt" ]; then
  echo "📄 Installing requirements..."
  pip install -r requirements.txt
else
  echo "⚠️  No requirements.txt found. Skipping package installation."
fi

echo "✅ Environment setup complete."