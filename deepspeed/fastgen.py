from mii import pipeline
pipe = pipeline("/home/cjl/llama/llama-2-7b-hf")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)