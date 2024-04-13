
from accelerate import Accelerator

from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer

from statistics import mean

import torch, time, json

accelerator = Accelerator()

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books

prompts_all=[

    "The King is dead. Long live the Queen.",

    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",

    "The story so far: in the beginning, the universe was created.",

    "It was a bright cold day in April, and the clocks were striking thirteen.",

    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",

    "The sweat wis lashing oafay Sick Boy; he wis trembling.",

    "124 was spiteful. Full of Baby's venom.",

    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",

    "I write this sitting in the kitchen sink.",

    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",

] #* 10

# load a base model and tokenizer

model_path="/home/cjl/llama/llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    device_map={"": accelerator.process_index},

    torch_dtype=torch.bfloat16,

)

tokenizer = AutoTokenizer.from_pretrained(model_path)   

# sync GPUs and start the timer

accelerator.wait_for_everyone()

start=time.time()

# divide the prompt list onto the available GPUs 

with accelerator.split_between_processes(prompts_all) as prompts:

    # print(prompts)

    # store output of generations in dict

    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference, prompt by prompt

    for prompt in prompts:

        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")

        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        # remove prompt from output 

        output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # store outputs and number of tokens in result{}

        results["outputs"].append( tokenizer.decode(output_tokenized) )

        results["num_tokens"] += len(output_tokenized)

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs

results_gathered=gather_object(results)

if accelerator.is_main_process:

    timediff=time.time()-start

    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")

    print("device_map is ",model.hf_device_map)

    print(results)



# auto
# tokens/sec: 22.0, time 44.367995500564575, total tokens 1000, total prompts 10
# device_map is  {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 1}
# [{'outputs': ['\nThe King is dead. Long live the Queen.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\n', 'They were sent to the country to stay with their eccentric uncle, who lived in a large house that had been in his family for hundreds of years.\nTheir uncle was a very strange man. He was tall and thin and had a long, hooked nose. He was always dressed in a long black cloak, and he had a long, white beard. He was very fond of children, and he was always telling them stories about the witches who lived in the', 'The universe was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, "Let there be light," and there was light. And God saw the light, that it was good; and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.\nAnd God said, "Let there', '\nWinston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.\nThe hallway smelt of boiled cabbage and old rag mats. At one end of it a colored poster, too large for indoor display, had been tacked to the wall. It', '\nIt is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.\n"It is a truth universally acknowledged, that a single man in possession of a good', '\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said.', "\n124 was spiteful. Full of Baby's venom.\n124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124", 'He was laying on his back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin in comparison to the rest of his bulk, flickered helplessly before his eyes.\n“What’s happened to me?” he thought. It was no dream.', 'I’m not sure why I’m here. I’m not sure why I’m writing this. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the', '\nI remember saying something like, "I feel a bit lightheaded; maybe you should drive..."\nAnd suddenly there was a terrible roar all around us and the sky was full of what looked like huge bats, all swooping and screeching and diving around the car, which was going about a hundred miles an hour with the top down to Las Vegas.\nAnd a voice was screaming: "Holy Jesus! What are these goddamn'], 'num_tokens': 1000}]

# "": accelerator.process_index
# tokens/sec: 29.0, time 34.17957544326782, total tokens 1000, total prompts 10
# device_map is  {'': 0}
# [{'outputs': ['\nThe King is dead. Long live the Queen.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\n', 'They were sent to the country to stay with their eccentric uncle, who lived in a large house that had been in his family for hundreds of years.\nTheir uncle was a very strange man. He was tall and thin and had a long, hooked nose. He was always dressed in a long black cloak, and he had a long, white beard. He was very fond of children, and he was always telling them stories about the witches who lived in the', 'The universe was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, "Let there be light," and there was light. And God saw the light, that it was good; and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.\nAnd God said, "Let there', '\nWinston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.\nThe hallway smelt of boiled cabbage and old rag mats. At one end of it a colored poster, too large for indoor display, had been tacked to the wall. It', '\nIt is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.\n"It is a truth universally acknowledged, that a single man in possession of a good', '\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said. "I\'m no\' gonnae tell ye."\n"I\'m no\' gonnae tell ye," he said.', "\n124 was spiteful. Full of Baby's venom.\n124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124 was spiteful. Full of Baby's venom. 124", 'He was laying on his back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin in comparison to the rest of his bulk, flickered helplessly before his eyes.\n“What’s happened to me?” he thought. It was no dream.', 'I’m not sure why I’m here. I’m not sure why I’m writing this. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the kitchen sink. I’m not sure why I’m writing this in the', '\nI remember saying something like, "I feel a bit lightheaded; maybe you should drive..."\nAnd suddenly there was a terrible roar all around us and the sky was full of what looked like huge bats, all swooping and screeching and diving around the car, which was going about a hundred miles an hour with the top down to Las Vegas.\nAnd a voice was screaming: "Holy Jesus! What are these goddamn'], 'num_tokens': 1000}]
