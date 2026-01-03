import torch
def generate(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
):

    model.eval()
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    generated = prompt_tokens.clone()

    assert generated.size(0) == 1, "generate() only supports batch_size=1"
    # Get model device from first parameter
    model_device = next(model.parameters()).device
    assert generated.device == model_device, f"Input device {generated.device} doesn't match model device {model_device}"
    assert temperature > 0.0
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(generated)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                indices_to_remove = cumsum_probs > top_p
                indices_to_remove[:, 0] = False
                sorted_probs[indices_to_remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
                assert torch.isfinite(probs).all()
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            yield next_token.item()

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

def inference(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu"
):
    encoded_prompt = tokenizer.encode(prompt)
    prompt_tokens = torch.tensor(encoded_prompt, dtype=torch.long, device=device)

    all_tokens = encoded_prompt.copy()

    for token in generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.vocabulary.get(b'<|endoftext|>')):
        all_tokens.append(token)
        yield tokenizer.decode(all_tokens)
