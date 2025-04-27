FROM ubuntu:22.04

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tehran

# Install dependencies, including texlive-xetex for xelatex
RUN apt-get update && apt-get install -y \
    wget \
    texlive \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

# Install newer Pandoc version (3.2) that supports --citeproc
RUN wget https://github.com/jgm/pandoc/releases/download/3.2/pandoc-3.2-linux-amd64.tar.gz \
    && tar -xzf pandoc-3.2-linux-amd64.tar.gz \
    && mv pandoc-3.2/bin/pandoc /usr/local/bin/ \
    && rm -rf pandoc-3.2 pandoc-3.2-linux-amd64.tar.gz

# Set working directory
WORKDIR /data

# Copy paper and bibliography
COPY publications/paper.md paper.bib ./

# Command to generate PDF
CMD ["pandoc", "paper.md", "-o", "paper.pdf", "--citeproc", "--bibliography=paper.bib", "--pdf-engine=xelatex"]