# CNPP

This package includes the implementation of our work **"Coupled Point Process-based Sequence Modeling for Privacy-preserving Network Alignment"**

# Dependencies

- Python 3.9.7
- PyTorch 1.11.0
- networkx
- scipy
- numpy 
- matplotlib

  

# Run the code

1. Generate event sequences according to the network structure (**simulator/simulator.py**)
2. Get an intensity-based prior of alignment matrix (**simulator/get_prior.py**)
3. Run our sequential behavior-driven network alignment method (**CNPP-SAHP/main.py** or **CNPP-THP/main.py**)