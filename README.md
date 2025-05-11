# Generics revisited: Analyzing generalizations in children's books and caregivers' speech

**Sunny Yu, Alvin W. M. Tan, Siying Zhang, Xuhui Miao, Riley Carlson, Tobias Gerstenberg, David Rose**

Presented at the 47th Annual Meeting of the Cognitive Science Society (2025; San Fransisco, CA).

[Link to paper](https://github.com/sunnyych/Children_Speech_Books/Generics_revisited_CogSci_paper.pdf)

```
@inproceedings{yu2025genericsrevisited,
  title = {Generics revisited: Analyzing generalizations in children's books and caregivers' speech},
  booktitle = {Proceedings of the 47th {Annual} {Conference} of the {Cognitive} {Science} {Society}},
  author = {Yu, Sunny, Tan, Alvin, Zhang, Siying, Miao, Xuhui, Carlson, Riley, Gerstenberg, Tobias, and Rose, David},
  year = {2025},
}
```

**Contents:**

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [CRediT author statement](#credit-author-statement)

## Overview

Generics, general statements about categories, are believed to transmit essentialist beliefs---the idea that things have a hidden true nature. Research suggests that people essentialize natural (biological and non-living) and social kinds, but not artifacts. Previous studies using small datasets found that generics are often used to describe animate beings in speech to children. Using a larger corpus of children's books and parent speech, we examined a wider range of kinds and generalizing statements (including habituals and universals). Our results show that generics are more likely used for biological kinds than artifacts and that their use increases in parent speech as children age. However, generics weren't more likely used for non-living or social kinds than artifacts. Habituals, at least in speech, were more likely used for social kinds than artifacts. Generalizing statements were more likely used for about non-living natural kinds than artifacts. These findings inform the debate over whether generics transmit essentialist beliefs.

## Repository structure

```
project_root/
├── code/
│   ├── python/
│   │   ├── analysis/
│   │   │   ├── analysis.ipynb
│   │   │   ├── analysis_parents.ipynb
│   │   │   └── filter_binary_classification.ipynb
│   │   ├── preprocess/
│   │   │   ├── NER_model_training/
│   │   │   ├── situation_entity_type_classification/
│   │   │   ├── dependency_parsing.py
│   │   │   └── merge_preprocess.ipynb
│   │   ├── classification/
│   │   │   └── binary_classification.py
│   │   ├── extraction/
│   │   │   └── quantified_statement_extraction.py
│   │   └── evaluation/
│   │       └── calculate_agreement.py
│
│   └── R/
│       ├── analysis/
│       │   ├── code/
│       │   │   ├── analysis.qmd
│       │   │   └── analysis_sunny.Rmd
│       ├── sample_data/
│       │   └── parsed_speech_sample.csv
│       └── child_speech_books.Rmd
│
├── data/
│   ├── dependency_parsing_data/
│   ├── eval/
│   └── examples/
│
├── figures/

```

- `/code`: code for training models as well as analysis + plotting code

  - `/python`: this folder contains python code for preprocessing data and the entire training-to-analysis pipeline
    - `/analysis`: this folder contains key analysis files for the data, where filter_binary_classification is used to filter out non-generic and non-habitual statements using an LLM-as-judge method.
    - `/preprocess`: this folder contains Python code used for fine-tuning BERT for object tagging (in the folder NER_model_training) and the code for situation entity type classification.
  - `/R`: this folder contains R code for plotting and performing statistical tests.

- `/data`: contains sample data used in the preprocessing and analysis files. Because of size limits and copy-right constraints we do not release the complete dataset but the dataset could be shared upon request. Please email syu03@stanford.edu if interested in the full dataset.

- `/figures`: folder containing all figures in the paper

## Set up

The project uses Python 3 (tested on 3.10). Also, R should be installed and added to the PATH. We recommend using conda to set up the analysis environment:

```
conda env create -f environment.yml
conda activate generics_revisited
```

## CRediT author statement

_What is a [CRediT author statement](https://www.elsevier.com/authors/policies-and-guidelines/credit-author-statement)?_

- **Sunny Yu:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Supervision, Project administration
- **Alvin Tan:** Methodology, Formal analysis, Writing - Original Draft, Writing - Review & Editing
- **Siying Zhang:** Resources, Data Curation
- **Xuhui Miao:** Software, Resources, Data Curation
- **Riley Carlson:** Resources, Data Curation
- **Tobias Gerstenberg:** Conceptualization, Methodology, Writing - Review & Editing, Supervision, Project administration, Funding acquisition
- **David Rose:** Conceptualization, Methodology, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Supervision, Project administration, Funding acquisition
