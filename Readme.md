# ComGPT Experiment Guide

## Overview

This guide explains how to reproduce the experimental results of ComGPT on both small and large datasets.

## Project Structure

project/
├── dataset/                  # Datasets
│   ├── football/             # Small network (with structure and ground-truth communities)
│   ├── dolphins/
│   ├── polbooks/
│   ├── amazon/               # Large network
│   └── dblp/
│
├── code/                     # Core code
│   ├── ComGPT on small datasets/       # ComGPT on small datasets (football, dolphins, polbooks)
│   ├── ComGPT on large datasets/       # ComGPT on large datasets (amazon, dblp)
│   └── GPTLCD/               # GPTLCD: Function package required by ComGPT
│
├── requirements.txt          # Dependency list
└── README.md                 # Project description

project/
├── dataset/ # Datasets
│ ├── football/ # Small network (with structure and ground-truth communities)
│ ├── dolphins/
│ ├── polbooks/
│ ├── amazon/ # Large network
│ └── dblp/
│
├── code/ # Core code
│ ├── small_datasets/ # ComGPT on small datasets (football, dolphins, polbooks)
│ ├── large_datasets/ # ComGPT on large datasets (amazon, dblp)
│ └── gptlcd/ # GPTLCD: Function package required by ComGPT
│
├── requirements.txt # Dependency list
└── README.md # Project description

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




If there are any bugs in the code, please contact us in time.

