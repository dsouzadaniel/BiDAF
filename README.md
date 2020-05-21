# Bi-Directional Attention Flow for Machine Comprehension
This is a :fire: PyTorch :fire: implementation of the seminal BiDAF Paper from Allen AI(***https://arxiv.org/pdf/1611.01603.pdf***) 

## Architecture

![BIDAF Architecture](https://raw.githubusercontent.com/dsouzadaniel/BiDAF/master/bidaf_model_architecture.png)

My modifications include :neckbeard: :rocket: :
* QA on a token level instead of a character level ( I use Spacy 2.1.9 to create the dataset from SQUAD )
* Swapping out GloVe for Elmo for improved performance 


## Dataset
I have uploaded a sample dataset from the SQUAD Dataset.
Download the complete dataset at [SQUAD QA Dataset](https://rajpurkar.github.io/SQuAD-explorer/)


## Demo
:bowtie: I have also implemented a Streamlit App to interact with the model! :bowtie:

![Streamlit App](https://raw.githubusercontent.com/dsouzadaniel/BiDAF/master/bidaf_model_app.png)


## Instructions 

#### To Install Requirements

> `pip install -r requirements.txt`

#### To Train a New Model

> ` python train.py`

#### To Run the Demo Streamlit App
 
> ` streamlit run app.py`

## Improvements
:v: Pull requests are welcome for any improvements/features :v:

