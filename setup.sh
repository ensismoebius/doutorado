#!/bin/bash

# Repository Setup Script
# Script de Configuração do Repositório
#
# This script helps set up the PhD research repository
# Este script ajuda a configurar o repositório de pesquisa de doutorado

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from repository root
if [ ! -f "README.md" ] || [ ! -d ".git" ]; then
    print_error "This script must be run from the repository root directory"
    exit 1
fi

print_info "Starting repository setup..."
echo ""

# 1. Check system requirements
print_info "Checking system requirements..."

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 found: $(command -v $1)"
        return 0
    else
        print_warning "$1 not found (optional for some components)"
        return 1
    fi
}

# Essential tools
MISSING_ESSENTIAL=0

if ! check_command "git"; then
    MISSING_ESSENTIAL=1
fi

if ! check_command "cmake"; then
    print_warning "CMake not found - required for C++ framework"
fi

if ! check_command "g++" && ! check_command "clang++"; then
    print_warning "No C++ compiler found - required for C++ framework"
fi

if ! check_command "python3"; then
    print_warning "Python 3 not found - required for notebooks"
fi

if [ $MISSING_ESSENTIAL -eq 1 ]; then
    print_error "Missing essential tools. Please install git."
    exit 1
fi

echo ""

# 2. Initialize git submodules if any
print_info "Checking git submodules..."
if [ -f ".gitmodules" ] && [ -s ".gitmodules" ]; then
    git submodule update --init --recursive
    print_success "Git submodules initialized"
else
    print_info "No git submodules to initialize"
fi

echo ""

# 3. Setup software/nn (C++ Neural Network Framework)
if [ -d "software/nn" ]; then
    print_info "Setting up Neural Network Framework (software/nn)..."
    
    read -p "Do you want to build the C++ framework now? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd software/nn
        
        print_info "Creating build directory..."
        mkdir -p build
        
        print_info "Running CMake configuration..."
        if cmake -S . -B build -DCMAKE_BUILD_TYPE=Release; then
            print_success "CMake configuration successful"
            
            print_info "Building the framework (this may take a while)..."
            # Detect number of processors (works on Linux and macOS)
            if command -v nproc &> /dev/null; then
                JOBS=$(nproc)
            elif command -v sysctl &> /dev/null; then
                JOBS=$(sysctl -n hw.ncpu)
            else
                JOBS=2
            fi
            if cmake --build build -- -j$JOBS; then
                print_success "Build successful!"
                
                read -p "Do you want to run tests? (y/n) " -n 1 -r
                echo
                
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    print_info "Running tests..."
                    if ctest --test-dir build --output-on-failure; then
                        print_success "All tests passed!"
                    else
                        print_warning "Some tests failed. Check output above."
                    fi
                fi
            else
                print_error "Build failed. Check errors above."
            fi
        else
            print_error "CMake configuration failed. Check errors above."
        fi
        
        cd ../..
    else
        print_info "Skipping C++ framework build"
    fi
fi

echo ""

# 4. Setup Python environment
print_info "Checking Python environment..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
    
    read -p "Do you want to create a virtual environment for Python? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created in ./venv"
        print_info "Activate it with: source venv/bin/activate"
        
        read -p "Do you want to install Jupyter and common packages? (y/n) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing packages..."
            source venv/bin/activate
            pip install --upgrade pip
            pip install jupyter numpy scipy matplotlib torch
            print_success "Packages installed"
            print_info "Start Jupyter with: jupyter notebook notebooks/"
        fi
    fi
else
    print_warning "Python 3 not found. Skipping Python setup."
fi

echo ""

# 5. Summary
print_success "Repository setup complete!"
echo ""
echo "==============================================="
echo "Next steps:"
echo "==============================================="
echo ""
echo "1. Review the main README.md for project overview"
echo "2. Check software/nn/README.md for detailed framework documentation"
echo "3. Review software/nn/TODO.md for experimental pipeline status"
echo "4. Explore notebooks/ directory for analysis examples"
echo "5. Check documentation/ for research materials"
echo ""

if [ -d "software/nn/build" ]; then
    echo "C++ Framework commands:"
    echo "  - Build: cd software/nn && cmake --build build"
    echo "  - Test:  cd software/nn && ctest --test-dir build"
    echo "  - Clean: rm -rf software/nn/build"
    echo ""
fi

if [ -d "venv" ]; then
    echo "Python environment:"
    echo "  - Activate: source venv/bin/activate"
    echo "  - Jupyter: jupyter notebook notebooks/"
    echo "  - Deactivate: deactivate"
    echo ""
fi

echo "For questions or issues, see CONTRIBUTING.md"
echo ""
print_success "Happy researching! / Boa pesquisa!"
