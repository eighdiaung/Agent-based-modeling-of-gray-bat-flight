# Agent-Based Modeling of Gray Bat Flight

This repository contains MATLAB scripts and data files for an agent-based model of gray bat flight behavior, with and without obstacles.

## MATLAB Scripts

- **model_NO.m**  
  MATLAB script for the model **without obstacles**.

- **model_O.m**  
  MATLAB script for the model **with obstacles**.

## Data Files - in TIDY format (Time | ID | X | Y | Z)

**Note on reference points:**  
- In `9_23.mat`, the *wall points* consist of 5 arbitrary reference points tracked along the edge of the left wall of the channel, where bats tend to fly in close proximity. These points are used as spatial references and are not bat trajectories.  
- In `9_24_obsref.mat`, the *obstacle points* are stored as a matrix of size N × 3, where N = 36 (top and bottom points for 18 obstacles) The points are ordered as:  
  (top obstacle 1, bottom obstacle 1, top obstacle 2, bottom obstacle 2, … , top obstacle 18, bottom obstacle 18),  with the three columns corresponding to the X, Y, and Z spatial dimensions.

- **9_23.mat**  
  Trajectories of **multiple bats without obstacles**.

- **9_23_singletons.mat**  
  Trajectories of a **single bat without obstacles**.

- **9_24_obsref.mat**  
  Trajectories of **multiple bats with obstacles**.

- **9_24_singletons_obsref.mat**  
  Trajectories of a **single bat with obstacles**.
