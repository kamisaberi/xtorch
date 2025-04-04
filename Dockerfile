FROM debian:bullseye-slim

LABEL maintainer="Kamal Saberi"
LABEL description="Build environment for compiling JOSS papers into PDF"

# Avoids interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-xetex \
    pandoc \
    make \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Default workdir
WORKDIR /paper

# Copy local files (e.g., paper.md, paper.bib, figures/) into the image
COPY . /paper

# Build command (optional - can also run manually)
CMD ["pandoc", "paper.md", "--from", "markdown", "--output", "paper.pdf", "--pdf-engine=xelatex", "--filter", "pandoc-citeproc", "--citeproc", "--bibliography=paper.bib"]
