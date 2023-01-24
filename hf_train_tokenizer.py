# Reference : https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/train_new_tokenizer.ipynb#scrollTo=jJLRnJVMcsIK 
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained(
  'huggingface-course/bert-base-uncased-tokenizer-without-normalizer'
)

# ---------- Samples -------------
text = "here is a sentence adapted to our tokenizer"
print(tokenizer.tokenize(text))

text = "the medical vocabulary is divided into many sub-token: paracetamol, phrayngitis"
print(tokenizer.tokenize(text))


# -------- Train a new tokenizer for code  -----------
from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("code_search_net", "python")
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
training_corpus = get_training_corpus()
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
new_tokenizer.save_pretrained("code-search-net-tokenizer")


example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
print(old_tokenizer.tokenize(example))
print(new_tokenizer.tokenize(example))
