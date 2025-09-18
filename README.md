# CE-GY-7893 Engineering Applications of Deep Learning Examples

A minimal setup for deep learning examples using Jupyter notebooks.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CE-GY-7893-Engineering-Applications-of-Deep-Learning-Examples

# Install dependencies
uv sync
```

### Usage

```bash
# Activate the virtual environment
source .venv/bin/activate

# Start Jupyter Lab
uv run jupyter lab

# Or start Jupyter Notebook
uv run jupyter notebook
```

### Project Structure

- `examples/` - Contains Jupyter notebooks with deep learning examples
- `pyproject.toml` - Project configuration and dependencies

### Jupytext

This project uses [Jupytext](https://jupytext.readthedocs.io/) to sync notebooks with Python files. This allows for better version control of notebook content.

To pair a notebook with a Python file:
```bash
uv run jupytext --set-formats ipynb,py:percent notebook.ipynb
```