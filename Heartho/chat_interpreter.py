from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaseModel:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()
    
    def generate(self, prompt, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class CustomAlgorithmCore:
    def __init__(self):
        self.algorithms = []

    def register_algorithm(self, name, fn):
        self.algorithms.append((name, fn))

    def run(self, input_data):
        results = {}
        for name, fn in self.algorithms:
            results[name] = fn(input_data)
        return results
    