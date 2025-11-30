from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

something_metric = GEval(
    name='Something',
    criteria = "Bruh",
    evaluation_steps = [],
)