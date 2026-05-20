# ComGPT Experiment Guide

## Overview

This guide explains how to reproduce the experimental results of ComGPT on both small and large datasets.

## Project Structure

# 📁 dataset/
- **football/** - football dataset
- **dolphins/** - dolphins dataset
- **polbooks/** - polbooks dataset 
- **amazon/** - amazon dataset
- **dblp/** - dblp dataset

# 📁 code/
- **ComGPT on small datasets/** - Implementation for small datasets (football, dolphins, polbooks)
- **ComGPT on large datasets/** - Implementation for large datasets (amazon, dblp)
- **GPTLCD/** - Core function package required by ComGPT

## Requirements

- Operating System: Windows 10
- Python: 3.8
- Python Packages:
  - matplotlib (3.7.5)
  - networkx (3.1)
  - numpy (1.24.4)
  - requests (2.32.3)

## Usage

### Small Datasets

To run ComGPT on small datasets (Football, Dolphins, Polbooks):

bash

```
python ComGPT_for_small_dataset.py
```



### Large Datasets

To run ComGPT on large datasets (Amazon, Dblp):

bash

```
python ComGPT_for_big_dataset.py
```

## Notes

- Ensure that all required Python packages are installed before running the code.
- The experimental results may vary depending on the LLM API settings and runtime environment.
- Some comparison algorithms may require additional configuration according to their respective instructions.



If there are any bugs in the code, please contact us in time.








