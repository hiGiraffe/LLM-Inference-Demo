from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is Sherry and I'm a 35-year-old woman from the United States. I've been a vegetarian for over 15 years and I'm passionate about promoting plant-based living for my health, the environment, and animal welfare. I've been following a mostly vegan lifestyle for the past 5 years and I love experimenting with new recipes and products in the plant-based space.",
]
# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enforce_eager=True)


def generate100(num=100):
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


def generate200(num=200):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=num, min_tokens=num)
    outputs = llm.generate(prompts, sampling_params)
    # outputs = llm.generate(prompts)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    generate100()
    #generate200()
