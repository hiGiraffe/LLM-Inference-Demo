
from accelerate import Accelerator

from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer

from statistics import mean

import torch, time, json

accelerator = Accelerator()

def write_pretty_json(file_path, data):

    import json

    with open(file_path, "w") as write_file:

        json.dump(data, write_file, indent=4)

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books

prompts_all=[
    "hello, here is",

    "The King is",

    "Once there were four children whose names",

    "The story so far: in the beginning, the universe was",

    "It was a bright cold day in April, and the clocks were",

    "It is a truth universally acknowledged, that a single man in possession of a good fortune",

    "The sweat wis lashing oafay Sick Boy; he",

    "124 was spiteful. Full of Baby's",

    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed",

    "I write this sitting in the kitchen",

    "We were somewhere around Barstow on the edge of the desert when the drugs began to",

] #* 10

# load a base model and tokenizer

model_path="/home/cjl/llama/llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    # device_map={"": accelerator.process_index},
    device_map="auto",

    torch_dtype=torch.bfloat16,

)

tokenizer = AutoTokenizer.from_pretrained(model_path)   

tokenizer.pad_token = tokenizer.eos_token

# batch, left pad (for inference), and tokenize

def prepare_prompts(prompts, tokenizer, batch_size=16):

    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)] 

    batches_tok=[]

    tokenizer.padding_side="left"     

    for prompt_batch in batches:

        batches_tok.append(

            tokenizer(

                prompt_batch, 

                return_tensors="pt", 

                padding='longest', 

                truncation=False, 

                pad_to_multiple_of=2,

                add_special_tokens=False).to("cuda") 

            )

    tokenizer.padding_side="right"

    return batches_tok

# sync GPUs and start the timer

accelerator.wait_for_everyone()   

# divide the prompt list onto the available GPUs 

# with accelerator.split_between_processes(prompts_all) as prompts:
if accelerator.is_main_process:
    start=time.time()

    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches

    prompt_batches=prepare_prompts(prompts_all, tokenizer, batch_size=16)

    print("\nprompt_batches\n",torch.cuda.current_device(),"\n",prompt_batches)

    for prompts_tokenized in prompt_batches:

        print("\nbegin prompts\n",torch.cuda.current_device(),"\n",prompts_tokenized)

        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=100)

        # remove prompt from gen. tokens

        # outputs_tokenized=[ tok_out[len(tok_in):] 

        #     for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

        # count and decode gen. tokens 

        num_tokens=sum([ len(t) for t in outputs_tokenized ])

        outputs=tokenizer.batch_decode(outputs_tokenized)

        # store in results{} to be gathered by accelerate

        results["outputs"].extend(outputs)

        results["num_tokens"] += num_tokens
        print("\noutput\n",torch.cuda.current_device(),"\n",outputs)

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs

# results_gathered=gather_object(results)

# if accelerator.is_main_process:

    timediff=time.time()-start

    num_tokens=sum([r["num_tokens"] for r in results ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

    print("\ndevice_map is\n",model.hf_device_map)

    print("\nresult is\n",results)