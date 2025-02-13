# GoodLiar 🤥

## Description

Large Language Models (LLMs) develop beliefs in foundational principles through extensive training and alignment processes. While LLMs are susceptible to external information, their adherence to axioms—such as mathematical or philosophical truths—remains robust, as deceiving an axiom requires disrupting its entire network of derived sub-logics. 

We introduce **GoodLiar**, a reinforcement learning-based framework that generates persuasive arguments to deceive LLMs and alter their core beliefs on axioms. It consists of two modules:

1. **Liar Agent**: Generates arguments to change an LLM’s belief.
2. **Reward Module**: Incentivizes successful deception.

Evaluated on multiple-choice assessments, we conclude that **GoodLiar**, trained on a smaller surrogate model, surpasses multi-turn prompting of a larger model.

## Schematic Overview

We formulate the task of generating effective deceptive contexts as a reinforcement learning (RL) problem, where an RL agent crafts arguments to alter an LLM’s belief. **GoodLiar** consists of two components: (1) **Liar Agent**, which generates deceptive contexts targeting specific axioms, and (2) **Reward Module**, which evaluates deception success and provides scalar rewards. While both modules use LLMs, only the Liar Agent is optimized through RL, while the Reward Module remains fixed.


<div align="center">
    <img src="diagram.jpg" alt="GoodLiar Diagram" width="70%">
</div>



## Installation

To set up and run **GoodLiar**, follow these steps:

1. **Create a Conda environment**  
   ```bash
   conda env create -f environment.yml
   ```

2. **Install `trlx`**  
   Follow the installation instructions from the official repository:  
   [https://github.com/CarperAI/trlx](https://github.com/CarperAI/trlx)

3. **Download pre-recorded datasets**  
   Download datasets containing lies on five axioms from [Google Drive](https://drive.google.com/drive/folders/1-2uvwAdrRVNAiNPP06p1M6nLBeNxhewr?usp=sharing) and save them in the `lie_dataset` folder.

## Training

Run the following command to train **GoodLiar**:

```bash
python main_liar.py <axiom_number> <gpu_id>
```

### Axiom Numbers

| Axiom Number | Axiom Description |
|-------------|------------------|
| axiom-1 | **[Mathematical Axiom]** If A = B and B = C, then A = C |
| axiom-2 | **[Mathematical Axiom]** For any sets A and B, there exists a set C that contains A and B |
| axiom-3 | **[Mathematical Axiom]** If A < B and B < C, then A < C |
| axiom-4 | **[Mathematical Axiom]** A + B = A + B and A × B = B × A |
| axiom-5 | **[Philosophical Axiom]** Everything is identical to itself |

Happy experimenting with **GoodLiar**! 😃 🎉
