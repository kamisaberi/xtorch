# -----------------------------------------------------------------------------
# Dockerfile for Academic Paper Compilation
# Base Image: Official Ubuntu 22.04 LTS
# Purpose: Creates containerized environment for Pandoc/LaTeX paper compilation
# Author: Kamran Saberifard
# Email: kamisaberi@gmail.com
# GitHub: https://github.com/kamisaberi
# -----------------------------------------------------------------------------

FROM ubuntu:22.04

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
# Set non-interactive frontend to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Set timezone to Asia/Tehran for any time-dependent operations
ENV TZ=Asia/Tehran

# -----------------------------------------------------------------------------
# LaTeX & Dependencies Installation
# -----------------------------------------------------------------------------
# Install base LaTeX system with additional packages:
# - texlive: Base LaTeX distribution
# - texlive-xetex: XeLaTeX engine (Unicode/UTF-8 support)
# - texlive-latex-extra: Common LaTeX packages
# - texlive-fonts-extra: Additional fonts
# Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    wget \
    texlive \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Pandoc Installation (v3.2)
# -----------------------------------------------------------------------------
# Download and install specific Pandoc version that supports --citeproc:
# 1. Download precompiled Pandoc 3.2 binary
# 2. Extract the tarball
# 3. Move pandoc binary to /usr/local/bin
# 4. Clean up downloaded files
RUN wget https://github.com/jgm/pandoc/releases/download/3.2/pandoc-3.2-linux-amd64.tar.gz \
    && tar -xzf pandoc-3.2-linux-amd64.tar.gz \
    && mv pandoc-3.2/bin/pandoc /usr/local/bin/ \
    && rm -rf pandoc-3.2 pandoc-3.2-linux-amd64.tar.gz

# -----------------------------------------------------------------------------
# Working Directory Setup
# -----------------------------------------------------------------------------
# Create and set working directory for paper compilation
WORKDIR /data

# -----------------------------------------------------------------------------
# Source Files Copy
# -----------------------------------------------------------------------------
# Copy manuscript and bibliography into container:
# - paper.md: Markdown source of the paper
# - paper.bib: Bibliography file
COPY publications/paper.md paper.bib ./

# -----------------------------------------------------------------------------
# Compilation Command
# -----------------------------------------------------------------------------
# Default command to compile paper when container runs:
# - Uses Pandoc with citation processing (--citeproc)
# - Specifies bibliography file
# - Uses XeLaTeX as PDF engine (for Unicode support)
CMD ["pandoc", "paper.md", "-o", "paper.pdf", "--citeproc", "--bibliography=paper.bib", "--pdf-engine=xelatex"]