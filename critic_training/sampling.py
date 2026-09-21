"""Batched categorical sampling with an independent RNG for each request."""
import torch
from transformers import LogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


class RequestSampler(LogitsProcessor):
    def __init__(self, seeds, device, temperature=.8, top_k=20, top_p=.95):
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.warpers = [TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k), TopPLogitsWarper(top_p)]

    def __call__(self, input_ids, scores):
        for warper in self.warpers:
            scores = warper(input_ids, scores)
        probabilities = torch.softmax(scores.float(), dim=-1)
        # Separate generators preserve each request's random stream across groups.
        tokens = torch.stack([torch.multinomial(p, 1, generator=g) for p, g in zip(probabilities, self.generators)])
        return torch.full_like(scores, -float('inf')).scatter_(1, tokens, 0.)


def generate(model, tokenizer, prompts, seeds, max_new_tokens):
    encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    sampler = RequestSampler(seeds, model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, logits_processor=[sampler],
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    return output[:, encoded['input_ids'].shape[1]:]
