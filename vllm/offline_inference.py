from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "1 Hello, my name is",
    "2 Hello, my name is",
    "3 Hello, my name is",
    "4 Hello, my name is",
    "5 Hello, my name is",
    "6 Hello, my name is",
    "7 Hello, my name is",
    "8 Hello, my name is",
    "9 Hello, my name is",
    "10 Hello, my name is",
]
# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enforce_eager=True)


def generate(num=100):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # Create a sampling params object.
    print("begin generate")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=num, min_tokens=num)
    outputs = llm.generate(prompts, sampling_params)
    # outputs = llm.generate(prompts)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    generate()
