# Question-Classification
Question Classification on the TREC dataset using HuggingFace's pre-trained DistilBert model (fine-tuned on TREC).

Python = 3.7.6

PyTorch

Dataset: https://cogcomp.seas.upenn.edu/Data/QA/QC/

1. Run the following command in your terminal:

pip install -r requirements.txt 

2. To fine-tune the model, run the following comand:

python Train_DistilBert.py

3. After training, in order to test the DistilBert model on your input run the following command:

python Test_DistilBert.py
