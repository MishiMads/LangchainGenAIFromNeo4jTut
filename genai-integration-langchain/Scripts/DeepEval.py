from deepeval.dataset import EvaluationDataset, Golden
from dotenv import  load_dotenv

load_dotenv()

# goldens are what makes up your dataset

goldens = [Golden(input="What's the weather like in SF?")]

# create dataset

dataset = EvaluationDataset(goldens=goldens)

# save to Confident AI

dataset.push(alias="YOUR-DATASET-ALIAS")