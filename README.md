# project_template

Project codebase template for 3dv research.

## Setup

```bash
# Clone the repository
git clone git@github.com:cxzhou35/project_template.git project_name

# Create a virtual environment (python >= 3.10)
conda create -n project_name python=3.10
conda activate project_name

# Install pytorch (cuda 12.1 etc.)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
cd project_name
pip install -r requirements.txt
```
