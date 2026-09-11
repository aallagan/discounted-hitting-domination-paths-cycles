# Exact and Fault-Tolerant Discounted Hitting Domination on Paths and Cycles
[![Reproducibility checks](https://github.com/aallagan/discounted-hitting-domination-paths-cycles/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/aallagan/discounted-hitting-domination-paths-cycles/actions/workflows/reproducibility.yml)
Reproducibility code accompanying the manuscript

**Exact and Fault-Tolerant Discounted Hitting Domination on Paths and Cycles**

by Julian D. Allagan, Kevin Pereyra, and William A. Massey.

## Contents

`reproduce_results.py` checks the principal finite-order formulas and
enumeration results of the paper on small instances.

The script verifies:

- exact discounted hitting domination numbers on paths and cycles;
- enumeration formulas for minimum placements;
- exact fault-tolerant values on paths and cycles;
- robust cycle enumeration;
- the path source-capacity formula; and
- the fault-tolerant complete-graph formula.

The verification computations use exact rational arithmetic where applicable.

## Requirements

- Python 3
- NetworkX 3.6.1
- Tested with Python 3.13 and NetworkX 3.6.1.
Install dependencies with:

```bash
pip install -r requirements.txt
