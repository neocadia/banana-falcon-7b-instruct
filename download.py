from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

if __name__ == "__main__":
    download_model()
