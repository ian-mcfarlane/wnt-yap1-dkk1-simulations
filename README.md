# wnt-yap1-dkk1-simulations
Code to produce simulations for "YAP1–DKK1–WNT axis links tissue mechanics to regeneration and defines the position of the gastric stem cell niche"

The code is structured as follows
* `multicellular_sims/run_model`: the main model simulation code that solves the governing equations of the model
* `multicellular_sims/run_simulations`: the driver code to run the experiments corresponding to Figures 6 and 7, and Supplementary Figure 3
* `multicellular_sims/voronoi_method.py`: helper to solve diffusion processes
* `parameters`: parameter `.yaml` files for simulations
* `plots`: plotting code for the model outputs related to Figures 6 and 7, and Supplementary Figure 3.
