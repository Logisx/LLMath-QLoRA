![Logo](https://github.com/Logisx/LLMath-QLoRA/blob/main/assets/logo-color-cropped.png?raw=true)

# :page_facing_up: Table of Contents 

- [:page\_facing\_up: Table of Contents](#page_facing_up-table-of-contents)
- [:rocket: LLM Instruction tuning for school math questions](#rocket-llm-instruction-tuning-for-school-math-questions)
  - [:bar\_chart: Model \& Dataset](#bar_chart-model--dataset)
  - [:toolbox: Tech Stack](#toolbox-tech-stack)
  - [:file\_folder: Project structure](#file_folder-project-structure)
  - [:computer: Run Locally](#computer-run-locally)
- [:world\_map: Roadmap](#world_map-roadmap)
- [⚖️ License](#️-license)
- [🔗 Links](#-links)
- [References \& Citations](#references--citations)
# :rocket: LLM Instruction tuning for school math questions

End-to-end MLOps project for LLM instruction finetuning based on PEFT & QLoRA to optimize the training.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-purple)
![JupyterLab](https://img.shields.io/badge/Jupyter%20Lab-Research-orange)
![Transformers](https://img.shields.io/badge/Transformers-NLP-amber)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![DVC](https://img.shields.io/badge/DVC-Version%20Control-ee4d5f)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![AWS](https://img.shields.io/badge/AWS-Cloud%20Deployment-FF9900)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Workflow-4D7A97)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 


![Demo](https://github.com/Logisx/IELTS-Grading/blob/main/assets/Demo.gif?raw=true)


## :bar_chart: Model & Dataset
**Base LLM**: [OpenLLaMA](https://huggingface.co/openlm-research/open_llama_3b_v2)\
**Dataset**: [Grade School Math Instructions Dataset](https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions)


## :toolbox: Tech Stack

- **NLP**: PyTorch, Jupyter Lab, Hugging Face Transformers, Accelerate, PEFT
- **Framework**: FastApi
- **Deployment**: Docker, Amazon Web Services (AWS), GitHub Actions
- **Version Control**: Git, DVC, GitHub
- **Testing**: REST client

## :file_folder: Project structure
Project structure template can be found [here](https://drivendata.github.io/cookiecutter-data-science/)
```
├── LICENSE
├── Makefile                  <- Makefile with commands like `make data` or `make train`
├── README.md                 <- The top-level README for developers using this project.
├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
│                                 generated with `pip freeze > requirements.txt`
|
├── config                    <- Stores pipelines' configuration files
|   ├── data-config.yaml
|   ├── model-config.yaml
|   └── model-parameters.yaml
|
├── data
│   ├── external              <- Data from third party sources.
│   ├── interim               <- Intermediate data that has been transformed.
│   ├── processed             <- The final, canonical data sets for modeling.
│   └── raw                   <- The original, immutable data dump.
│
├── assets                    <- Store public assets for readme file
├── docs                      <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                    <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                 <- Jupyter notebooks for research.
│
├── setup.py                  <- Make this project pip installable with `pip install -e`
├── src                       <- Source code for use in this project.
│   ├── __init__.py           <- Makes src a Python module
│   │
|   ├── logging               <- Define loggers for the app
|   ├── utils
|   |   ├── __init__.py
|   |   └── common.py         <- Functions for common utilities
|   |
│   ├── data                  <- Scripts to download or generate data
|   |   ├── components        <- Classes for pipelines
|   |   ├── pipeline          <- Scripts for data aggregation
|   |   ├── configuration.py  <- Class to manage config files
|   |   ├── entity.py         <- Stores configuration dataclasses
│   │   └── make_dataset.py   <- Script to run data pipelines
│   │
│   └── models                <- Scripts to train models and then use trained models to make
│       │                         predictions
|       ├── components        <- Classes for pipelines
|       ├── pipeline          <- Scripts for data aggregation
|       ├── configuration.py  <- Class to manage config files
|       ├── entity.py         <- Stores configuration dataclasses
│       ├── predict_model.py  <- Script to run prediction pipeline
│       └── train_model.py    <- Script to run model pipelines
│
├── main.py                   <- Script to run model training pipeline
├── app.py                    <- Script to start FastApi app
|
├── .env.example              <- example .env structure
├── Dockerfile                <- configurates Docker container image
├── .github
|   └── workflows
|       └── main.yaml         <- CI/CD config 
|
├── .gitignore                <- specify files to be ignored by git
├── .dvcignore                <- specify files to be ignored by dvc
|
├── .dvc                      <- dvc config 
├── dvc.lock                  <- store dvc tracked information
└── dvc.yaml                  <- specify pipeline version control
```

## :computer: Run Locally

1. Clone the project

```bash
  git clone https://github.com/Logisx/LLMath-QLoRA
```

2. Go to the project directory

```bash
  cd my-project
```

3. Install dependencies

```bash
  pip install -r requirements.txt
```

4. Start the app

```bash
  python app.py
```

# :world_map: Roadmap

1. **Testing features**: Develop unit tests and integrations test.
2. **Data collection**: Aggregate more data to improve accuracy.
3. **Educational insights feature**: Along with the score, the application will offer insights and suggestions for improvement, making it a valuable educational tool for those looking to enhance their writing skills.


# ⚖️ License

[MIT](https://github.com/Logisx/DeepEssay/blob/main/LICENSE)


# 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aleksandrshishkov)

# References & Citations