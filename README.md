# Master Thesis: Time-Efficient Value Iteration for Markov Decision Processes
This repository contains the source code used for my master thesis "Time-Efficient Value Iteration for Markov Decision Processes"

## How the thesis plots were generated 

### figures 6.2(a), 6.2(b), 6.3(a), 6.3(b): 
- Modelname: MacBook Pro Model-id: MacBookPro11,5 Processorname: Quad-Core Intel Core i7 Processorspeed: 2,5 GHz Antal processors: 1 number of cores in total: 4 L2-buffer (pr. Core): 256 kB L3-buffer: 6 MB Hyper-Threading-teknologi: Slået til Hukommelse: 16 GB

### All other figures
- Skadi server (39 cores), info about os in `osversion.txt` and complete processor info in `cpuinfo.txt` (very verbose)

## How to run experiments 
```
g++ -pthread -std=c++17 -o algo_test *.cpp && ./algo_test
````
- Threading were only used to run experiments in parallel (on a single core each), not within each experiment/algorithm I use c++17 features (structural bindings etc.) so this flag is necessary.
- experiments are all defined in: experiments.cpp (and method declerations in .h file) experiments are all run through: main.cpp
