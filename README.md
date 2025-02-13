# GoodLiar

Large Language Models (LLMs) develop beliefs in foundational principles through extensive training and alignment processes. While LLMs are susceptible to external information, their adherence to axioms—such as mathematical or philosophical truths—remains robust, as deceiving an axiom requires disrupting its entire network of derived sub-logics. 

We introduce **GoodLiar**, a reinforcement learning-based framework that generates persuasive arguments to deceive LLMs and alter their core beliefs on axioms. It consists of two modules: 
1. **Liar Agent**, which generates arguments to change an LLM’s belief.
2. **Reward Module**, which incentivizes successful deception.

Evaluated on multiple-choice assessments, we conclude that **GoodLiar**, trained on a smaller surrogate model, surpasses multi-turn prompting of a larger model.

## Schematic Overview

![GoodLiar Diagram](diagram.png)

## Installation Instructions

### Step 1: Create Code Environment
Create the Conda environment using the following command:

```bash
conda env create -f environment.yml
