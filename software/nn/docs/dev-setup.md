# Development Setup Guide for Linux Systems

This guide provides recommended tools, packages, and configurations for a Computer Science researcher on a Linux system. The examples are tailored to **Arch Linux** but translate easily to Debian/Ubuntu or other distributions.

## Table of Contents

1. [System Configuration](#system-configuration)
2. [Development Toolchain](#development-toolchain)
3. [Research & Reproducibility](#research--reproducibility)
4. [Libraries & Performance](#libraries--performance)
5. [Testing & CI](#testing--ci)
6. [Productivity Tools](#productivity-tools)
7. [Data & Visualization](#data--visualization)
8. [Security & Backup](#security--backup)
9. [Paper Writing](#paper-writing)
10. [Quick Start Checklist](#quick-start-checklist)

---

## System Configuration

### Automatic Time Sync & Updates

```bash
# Enable systemd-timesyncd for NTP
sudo systemctl enable --now systemd-timesyncd

# Check time sync status
timedatectl status

# Enable automatic security updates (Arch Linux)
sudo pacman -S --needed pacman-contrib
# Create /etc/pacman.d/hooks/auto-update.hook if desired
```

### Package Management

Install an AUR helper for streamlined package installation:

```bash
# For Arch Linux: install yay
sudo pacman -S yay

# Usage: yay <package-name>  # pulls from AUR and official repos
```

### CPU Governor & System Tuning

```bash
# Check current CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# For sustained workloads, prefer 'performance' or 'schedutil'
# Set in /etc/default/cpupower or via systemd service (if cpupower package is installed)

# Adjust swappiness for reduced swapping (lower = less swap use)
# Edit /etc/sysctl.conf:
#   vm.swappiness=10

# Increase ulimits for core dumps
ulimit -c unlimited
# Make permanent in ~/.bashrc or ~/.zshrc:
#   ulimit -c unlimited
```

---

## Development Toolchain

### Core Build Tools

```bash
# Install essential compilers and build system
sudo pacman -S --needed \
  gcc \
  clang \
  cmake \
  ninja \
  make \
  base-devel \
  ccache

# (Optional) Install distcc for distributed compilation
sudo pacman -S distcc

# Verify installation
gcc --version
clang --version
cmake --version
ninja --version
```

### Linters & Static Analysis

```bash
# Install clang-tidy, cppcheck, and formatting tools
sudo pacman -S --needed \
  clang-tools-extra \
  cppcheck \
  shfmt

# Optional: install shellcheck for shell script linting
sudo pacman -S shellcheck

# Set up pre-commit hooks (Python tool, works on any project)
pip install pre-commit
cd /path/to/doutorado/software/nn
pre-commit install
# Create .pre-commit-config.yaml with clang-format and clang-tidy checks (optional)
```

### Debuggers & Profilers

```bash
# Install GDB, LLDB, Valgrind, and perf
sudo pacman -S --needed \
  gdb \
  lldb \
  valgrind \
  linux-tools

# Optional: kcachegrind for call graph visualization
sudo pacman -S kcachegrind

# Verify debuggers
gdb --version
lldb --version

# Enable core dumps for debugging crashes
ulimit -c unlimited
echo "kernel.core_pattern=/tmp/core-%e-%s-%u-%g-%p-%t" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Configure ccache

```bash
# Create symlinks so ccache wraps the compilers
sudo ln -s /usr/bin/ccache /usr/local/bin/gcc
sudo ln -s /usr/bin/ccache /usr/local/bin/g++
sudo ln -s /usr/bin/ccache /usr/local/bin/clang
sudo ln -s /usr/bin/ccache /usr/local/bin/clang++

# Verify
which gcc  # should return /usr/local/bin/gcc

# Monitor cache stats
ccache -s
```

---

## Research & Reproducibility

### Python Environments

```bash
# Install conda (Miniconda or Mambaforge recommended)
yay -S mambaforge  # or miniconda3

# Initialize conda
conda init zsh  # (or bash)

# Create a project environment
conda create -n doutorado-env python=3.11 pip
conda activate doutorado-env

# Pin dependencies in requirements.txt or environment.yml
# requirements.txt example:
cat > requirements.txt << 'EOF'
numpy==1.24.3
scipy==1.11.0
matplotlib==3.7.1
pandas==2.0.2
scikit-learn==1.3.0
jupyterlab==4.0.0
EOF

pip install -r requirements.txt

# Freeze exact environment
pip freeze > requirements-lock.txt
conda env export > environment-lock.yml
```

### Experiment Tracking

```bash
# Install MLflow or Weights & Biases for experiment tracking
pip install mlflow  # or wandb, sacred, neptune

# MLflow example: log experiments, metrics, artifacts
python -c "
import mlflow
mlflow.set_experiment('my_exp')
with mlflow.start_run():
    mlflow.log_param('lr', 0.001)
    mlflow.log_metric('loss', 0.5)
    mlflow.log_artifact('model.pkl')
"

# View UI: mlflow ui
```

### Data Versioning

```bash
# Install git-lfs for large files
sudo pacman -S git-lfs
git lfs install

# Track data files
git add data/dataset.npy.dvc
```

### Containers (Optional but Recommended)

```bash
# Install Docker (Arch Linux)
sudo pacman -S docker
sudo systemctl enable --now docker

# Add user to docker group to avoid sudo
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect

# Or install apptainer (Singularity)
sudo pacman -S apptainer

# Example Dockerfile for this project
cat > Dockerfile << 'EOF'
FROM archlinux:latest
RUN pacman -Syu --noconfirm && pacman -S --needed --noconfirm base-devel cmake ninja eigen fftw
COPY . /src
WORKDIR /src
RUN mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && ninja
EOF

docker build -t doutorado:latest .
```

---

## Libraries & Performance

### OpenMP Configuration

```bash
# OpenMP is usually installed with gcc/clang. Verify:
gcc -fopenmp -dumpversion

# Set thread count for reproducibility
export OMP_NUM_THREADS=4

# For nested parallelism (use sparingly):
export OMP_NESTED=FALSE
```

### FFTW3 (Already Vendored in CMake)

The project's CMake already builds FFTW3 with `--enable-openmp --enable-threads`. Verify in build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
# Check that FFTW3 was found/built
grep -i fftw build/CMakeCache.txt
```

### Eigen (Already Vendored)

Eigen is header-only and already included. For best performance, compile with `-DCMAKE_CXX_FLAGS="-O3 -march=native"` and enable OpenMP:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -fopenmp"
```
---

## Testing & CI

### GoogleTest (Already Configured)

```bash
# Build and run tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j$(nproc)
ctest --test-dir build --output-on-failure -j4

# Run a specific test
ctest --test-dir build -R "tensor" --verbose
```

### Enable Sanitizers for Debug Builds

```bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer"

cmake --build build-asan -- -j$(nproc)
ctest --test-dir build-asan --output-on-failure
```

### GitHub Actions CI (Already Present)

Verify `.github/workflows/build-and-test.yml` exists and runs on push/PR. Update if needed for new tests.

---

## Productivity Tools

### VS Code Setup (Recommended)

```bash
# Install VS Code from AUR
yay -S visual-studio-code-bin

# Essential extensions for C++ development:
# - C/C++ (ms-vscode.cpptools)
# - clangd (llvm-vs-code-extensions.vscode-clangd)
# - CMake Tools (ms-vscode.cmake-tools)
# - GitLens (eamodio.gitlens)
# - LaTeX Workshop (james-yu.latex-workshop) — for paper writing

# Settings hint: enable clangd, set CMake kit to your compiler, use compile_commands.json
```

### Terminal & Session Management

```bash
# Install tmux or screen for long-running experiments
sudo pacman -S tmux

# Install mosh for robust remote connections
sudo pacman -S mosh

# Example tmux session for an experiment
tmux new-session -d -s exp01 -c /path/to/project
tmux send-keys -t exp01 "./run_experiment.sh" Enter
tmux attach -t exp01  # reattach later
```

### Jupyter for Exploratory Work

```bash
# Install JupyterLab in conda environment
conda activate doutorado-env
pip install jupyterlab notebook

# Optional: C++ kernel (xeus-cling)
pip install xeus-cling

# Start Jupyter
jupyter lab --ip=0.0.0.0 --no-browser  # accessible remotely
```

### Documentation Tools

```bash
# Install Sphinx for API docs, Doxygen for C++ docs
sudo pacman -S sphinx doxygen

# Install Pandoc for format conversion (LaTeX ↔ Markdown)
sudo pacman -S pandoc

# Example: generate Doxygen docs
doxygen Doxyfile
```

---

## Data & Visualization

### Python Data Stack

```bash
# Already installed via conda/pip, but listed for clarity:
pip install --upgrade \
  numpy \
  scipy \
  matplotlib \
  seaborn \
  pandas \
  scikit-learn \
  jupyter

# Optional: plotly for interactive plots
pip install plotly
```

### Matplotlib from C++ (Already Vendored)

The project includes `matplotlib-cpp`. Example usage:

```cpp
#include "matplotlibcpp.h"
#include <vector>

namespace plt = matplotlibcpp;

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {1, 4, 9, 16, 25};
    plt::plot(x, y);
    plt::show();
    return 0;
}
```

---

## Security & Backup

### SSH Keys

```bash
# Generate ed25519 key (recommended over RSA)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "researcher@work"

# Set restrictive permissions
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Enable ssh-agent for session key management
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Add to GitHub via Settings → SSH Keys
cat ~/.ssh/id_ed25519.pub  # copy and paste into GitHub
```

### Password Manager & Secrets

```bash
# Install pass (password manager)
sudo pacman -S pass

# Or use built-in OS keyring
# Store sensitive info in pass, never in scripts or git:
pass init <your-gpg-key>
pass insert work/github-token
# Later retrieve: pass work/github-token | xclip -selection clipboard
```

### Automated Backups

```bash
# Install borg (deduplicating backup)
sudo pacman -S borgbackup

# Initialize a repository
borg init --encryption=repokey /mnt/backup/doutorado

# Backup important directories
borg create /mnt/backup/doutorado::data-backup \
  /home/user/Repos/doutorado/software/nn/data \
  /home/user/notebooks

# Schedule with cron
# Edit crontab: crontab -e
# Add: 0 2 * * * borg create /mnt/backup/doutorado::backup-$(date +\%Y\%m\%d) ...
```

### 2FA Setup

```bash
# Enable 2FA on GitHub
# 1. Go to Settings → Password and authentication → Two-factor authentication
# 2. Scan QR code with Authenticator app (Authy, Google Authenticator, KeePass, etc.)
# 3. Save recovery codes in a secure location (encrypted USB, password manager)

# Install an offline TOTP app on your phone for GitHub, email, banking
```

---

## Paper Writing

### LaTeX Environment

```bash
# Install TeX Live (full distribution)
sudo pacman -S texlive-most texlive-lang

# Or install minimal and add packages as needed
sudo pacman -S texlive-core

# Verify
pdflatex --version
```

### Citation Management

```bash
# Install Zotero for bibliography management
yay -S zotero

# Export bibliography to BibTeX format
# In Zotero: Right-click collection → Export Collection → BibTeX

# Use in LaTeX preamble:
# \bibliography{refs.bib}
# \bibliographystyle{plain}

# Or use biblatex for more control:
% In preamble:
% \usepackage[backend=biber, style=numeric]{biblatex}
% \addbibresource{refs.bib}
% In document: \printbibliography
```

### Version Control for Papers

```bash
# Keep papers and supplementary materials in git
git add paper.tex supplementary_material.pdf figure_generation_script.py

# Use git to track changes
git log --oneline paper.tex

# Create releases for submitted versions
git tag -a v1.0-submitted -m "Paper submitted to JMLR"
```

---

## Quick Start Checklist

Copy and adapt this checklist for your system:

- [ ] **System**: Enable automatic time sync and updates
- [ ] **Build Tools**: Install gcc, clang, cmake, ninja, ccache
- [ ] **Linters**: Install clang-tidy, cppcheck, shfmt
- [ ] **Debuggers**: Install gdb, lldb, valgrind; enable core dumps
- [ ] **Python**: Create conda/venv environment; pin dependencies
- [ ] **Experiment Tracking**: Install MLflow or wandb
- [ ] **Data Versioning**: Install git-lfs or dvc
- [ ] **Containers**: (Optional) Install docker or apptainer
- [ ] **Editor**: Set up VS Code with clangd, CMake Tools
- [ ] **Terminal**: Install tmux or screen for remote work
- [ ] **Jupyter**: Install JupyterLab for exploratory analysis
- [ ] **Docs**: Install Sphinx, Doxygen, Pandoc
- [ ] **SSH**: Generate ed25519 keys; add to GitHub
- [ ] **Secrets**: Set up pass or OS keyring for credentials
- [ ] **Backup**: Configure borg or restic; test restore
- [ ] **2FA**: Enable on GitHub and critical accounts
- [ ] **LaTeX**: Install TeX Live; set up Zotero for citations
- [ ] **Build & Test**: Run `cmake`, `ninja`, `ctest` to verify everything works

---

## Further Resources

- **CMake**: https://cmake.org/cmake/help/latest/
- **Eigen**: https://eigen.tuxfamily.org/
- **FFTW3**: https://www.fftw.org/
- **Google Test**: https://google.github.io/googletest/
- **GitHub Actions**: https://docs.github.com/en/actions
- **Arch Linux Handbook**: https://wiki.archlinux.org/
- **Conda Documentation**: https://docs.conda.io/
- **MLflow**: https://mlflow.org/

---

**Last Updated**: December 11, 2025

For questions or updates to this guide, refer to `.github/copilot-instructions.md` and the main `README.md`.
