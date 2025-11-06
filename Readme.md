# Robust Gradient-Based Optimization Design of Laminar Wings Based on Manifold Learning

This folder contains the accompanying code for the paper titled **"Robust Gradient-Based Optimization Design of Laminar Wings Based on Manifold Learning"**. It can be used to verify the numerical examples presented in the paper.

## Program Workflow (4 Steps)

### Step 0: Generate Perturbation Variables

Run `LHS_sample.py` to generate perturbation variables.

### Step 1: Establish and Preprocess Airfoil Database

1. Execute `GeometryWarp_smooth.py` to generate the corresponding airfoil database.
2. Run `Sort_Slice.py` to preprocess the airfoil database, preparing it for subsequent manifold learning.

### Step 2: Perform Manifold Learning

Execute `RAE2822_pga.py` to conduct manifold learning on the preprocessed airfoil database.

### Step 3: Conduct Optimization

Run either `RAE2822Tran_Grassmannian_SubSonic.py` or `RAE2822Tran_Grassmannian_TranSonic.py` to optimize the data after manifold learning.
