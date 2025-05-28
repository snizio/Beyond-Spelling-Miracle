# Beyond the Spelling Miracle:Investigating Substring Awareness in Character-Blind Language Models

This is the public repository for our paper: *Beyond the Spelling Miracle: Investigating Substring Awareness in Character-Blind Language Models*, C. Ciaccio, M. Sartor, A. Miaschi, F. Dell'Orletta (ACL 2025). 

The repository contains the related code and configurations used to run our experiments for assessing character competence in character-blind language models. Specifically, the **config_files** folder contains the .json files with the corresponding arguments for each Pythia model; the file **pythia_train.py** contains the code to run our finetuning experiments; **dataset_generator.py** is the script to build our dataset from MorphoLex.

If you use any of the following contents for your work, we kindly ask you to cite our paper:

```bibtex
@inproceedings{
}
```

> **Abstract:** Correctly identifying characters and substrings of words should be a basic but essential ability of any Language Model that aims to proficiently understand and produce language. Despite so, the majority of Pre-trained Language Models (PLMs) are "character-blind" and struggle in spelling tasks, although they still seem to acquire some character knowledge during pre-training, a phenomenon dubbed \textit{Spelling Miracle}. 
To shed light on this phenomenon, we systematically evaluate a range of PLMs with different parameter sizes using a controlled binary substring identification task. Through a series of experiments, we propose the first comprehensive investigation on where, when, and how PLMs develop awareness of characters and substrings, with a particular linguistic focus on morphemic units such as prefixes, suffixes, and roots.