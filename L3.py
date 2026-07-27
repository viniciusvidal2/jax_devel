#!/usr/bin/env python
# coding: utf-8

# # L3: Data Loading with Grain

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[ ]:


import jax
import jax.numpy as jnp

import grain.python as grain

import tiktoken
from pathlib import Path
from helper import load_stories_from_file


# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# ## Dataset

# In[ ]:


file_path = Path("TinyStories-1000.txt")

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    data = f.read()
    stories = data.split('<|endoftext|>')

    print("First story (300 chars):\n")
    story = stories[0]
    print(story.strip()[:300], "...")

    print(f"\nTotal number of stories: {len(stories) - 1:,}")


# ## Tokenizer

# In[ ]:


tokenizer = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size: {tokenizer.n_vocab:,}")
print(f"Special tokens: {tokenizer.special_tokens_set}")


# ## Dataset Class

# In[ ]:


class StoryDataset:
    
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>', \
                        allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, 
                                       allowed_special={'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]
            
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


# ## Build a data iterator

# In[ ]:


shuffled_sampler = grain.IndexSampler(
    num_records=10,
    shuffle=True,
    seed=42,
    shard_options=grain.NoSharding(),    
    num_epochs=1
)

def print_sampler_example(sampler, name):
    print(f"\n{name}")
    for i, idx in enumerate(sampler):
        print(f"Record {i}: {idx}")

print_sampler_example(shuffled_sampler, "Shuffled sampler")


# ## Batch sequences into fixed-shape arrays

# In[ ]:


batch_op_keep = grain.Batch(
    batch_size=32,
    drop_remainder=False
)


# ## Build a data loader

# In[ ]:


def create_dataloader(
    stories,
    tokenizer,
    maxlen,
    batch_size,
    shuffle = False,
    num_epochs = 1,
    seed = 42,
    worker_count = 0
):
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    sampler = grain.IndexSampler(
        num_records=len(dataset), # 1,000 stories for this dataset
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=worker_count
    )
    
    return dataloader, estimated_batches


# In[ ]:


stories = load_stories_from_file(
    "TinyStories-1000.txt", 
    max_stories=100
)


# In[ ]:


stories[0]


# In[ ]:


dataloader, batches_per_epoch = create_dataloader(
    stories=stories,
    tokenizer=tokenizer,
    maxlen=128,
    batch_size=32,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0  # Single process for experimentation
)

print(f"\nDataLoader created successfully:")
print(f"Will produce {batches_per_epoch} batches per epoch")


# ## Fetching data

# In[ ]:


next(iter(dataloader))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




