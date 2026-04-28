import torch
import torch.nn as nn
import torch.nn.functional as F

# load the data from file
with open("data.txt", "r") as f:
    text = f.read()  # read entire content in string

# set objects are unordered and you cannot sort them or index them directly, if not done sorted is anyway returning a sorted list
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Build the 'string-to-integer' (stoi) dictionary
# enumerate(chars) yields pairs of (index, character) like (0, 'a'), (1, 'b')
# 'ch:i' sets the character as the Key and the index as the Value
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define a function to convert a string (text) into a list of numbers


def encode(s):
    # This list comprehension loops through every character 'c' in the input string 's'
    # It looks up each character in the 'stoi' dictionary to get its unique number
    return [stoi[c] for c in s]

# Define a function to convert a list of numbers back into a string


def decode(l):
    # 1. [itos[i] for i in l] creates a list of characters by looking up each number 'i'
    # 2. ''.join(...) takes that list of characters and glues them together into one string
    return ''.join([itos[i] for i in l])


# A tensor = multi-dimensional array (like NumPy array)
# Why tensors?
# Fast computation
# GPU support
# Required for neural networks
#  Converts that list into a PyTorch "Tensor" object.
data = torch.tensor(encode(text), dtype=torch.long)

# hyperparameters are settings that define how your GPT model will learn and how big its "brain" will be.
block_size = 8  # This is the context window. It means the model only looks at the last 8 characters to predict the 9th one. If you want it to remember long sentences, you’d eventually increase this.
batch_size = 16  # The model doesn't look at one example at a time; it looks at 16 chunks of text simultaneously in every training step to speed things up
# This is how many numbers the model uses to represent a single character. Instead of just "4", it represents the letter 'h' as a list of 32 different numbers to capture its meaning.
embedding_dim = 32
# This is Multi-Head Attention. It means the model looks at the text through 4 different "sets of eyes" simultaneously (one might look for grammar, another for rhyming, etc.).
num_heads = 4
# This is how many Transformer blocks are stacked on top of each other. More layers usually mean a smarter model but a slower training time.
num_layers = 2
# (0.001): This is how big of a "step" the model takes when correcting its mistakes. Too big, and it crashes; too small, and it takes forever to learn
learning_rate = 1e-3
# This is how many times the model will go through the training cycle. 2,000 rounds of practice.
epochs = 2000


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size))
    # For every random start index (i), we grab a chunk of 8 (block_size) characters
    x = torch.stack([data[i:i+block_size] for i in ix])
    # torch.stack = This takes those 16 separate chunks and "stacks" them on top of each other into a single matrix (a 16x8 grid).
    # This is the "answer key." It grabs the same chunk but shifted one position to the right. Why shift? Because GPT is a predictor. If the input (x) is [h, e, l, l], the target (y) should be [e, l, l, o]. It teaches the model: "When you see 'h', the next letter is 'e'."
    y = torch.stack([data[i+block_size+1] for i in ix])
    return x, y

# Self attention implementation
# This is the "Communication Layer." Self-attention is how every character in your block_size (8 characters) looks at the others to figure out which ones are important for predicting the next letter
# 1. The Three Roles (Query, Key, Value)
# In the __init__, we create three linear layers. Every character gets three vectors:
# Query (Q): "What am I looking for?" (The search term).
# Key (K): "What do I contain?" (The profile description).
# Value (V): "If I am important, what information do I share?" (The actual content).


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super.__init__()  # It tells Python to connect your SelfAttention class to all the built-in features of nn.Module
        # If your "brain" size is 32 and you have 4 heads, each head gets a smaller chunk of 8. Why? It’s more efficient. It’s like splitting a 32-person meeting into 4 specialized teams of 8.
        self.heads = heads
        self.head_dim = embed_size // heads

        # If you used one big layer, the model would only be able to focus on one relationship at a time. By splitting it into heads, you're forcing the model to learn multiple things at once (like grammar AND meaning).

        # These are Trainable Weights.
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # After you split the work into 4 heads and get 4 different results, you need a way to merge them back together. This layer "mixes" the insights from all heads back into a single 32-dimension representation.
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # This grabs the dimensions of the input data: Batch size (number of samples), Time/Sequence length (number of tokens), and Channels/Embedding size (vector size per token).
        B, T, C = x.shape

        # The input x is passed through linear layers (defined in __init__) to create the Query, Key, and Value matrices.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # .view(B, T, self.heads, self.head_dim)==> The input embedding dimension (C) is split into smaller, independent "heads"
        # .transpose(1, 2): The dimensions are rearranged to (B, Heads, T, Head_dim).
        # Why? This puts the Heads dimension before Time (T), allowing PyTorch to perform matrix multiplication on all heads independently and simultaneously.

        Q = Q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Q @ K.transpose(-2, -1): This is matrix multiplication (dot-product) between Queries and Keys, determining the affinity between every token and every other token.
        attn = (Q @ K.transpose(-2, 
        -1)) / (self.head_dim ** 0.5) # Scaling. The dot products are divided by the square root of the head dimension to keep gradients stable.

        # Converts the raw scores into probabilities (summing to 1) along the last dimension, determining how much focus one token places on others.
        attn = F.sofmax(attn, dim=-1)

        # The attention probabilities are multiplied by the Value () matrix.
        # Result: Each token's new representation is a weighted sum of the values of other tokens based on the attention scores
        out = attn @ V
        
        # Reverses the earlier transpose, moving the heads back next to the Time dimension.
        # .contiguous(): Ensures the tensor is stored continuously in memory, which is necessary after a transpose.
        # .view(B, T, C): Concatenates all the heads back together, reshaping the (B, T, Heads, Head_dim) tensor back to the original (B, T, C) format.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()

        self.attn = SelfAttention(embed_size, heads) # Plugs in the SelfAttention class you just built. This is the communication phase.
        self.ln1 = nn.LayerNorm(embed_size) # These are like "volume knobs." They normalize the numbers so no single neuron gets too loud or too quiet. This makes training much more stable.
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),  #  It expands the data to a larger space (32 becomes 128) to give it more "room" to process complex patterns.
            nn.ReLU(), # Rectified Linear Unit=  A simple filter that turns negative numbers to zero. This adds "non-linearity," which is a fancy way of saying it lets the model learn complex logic instead of just simple math.
            nn.Linear(4 * embed_size, embed_size) # It shrinks the data back down to its original size (32).
        )  # After communicating, every character needs to "think" individually.
        self.ln2 = nn.LayerNorm(embed_size)

        def forward(self, x):
            # These are called Residual Connections (or Skip Connections).
            x = x + self.attn(self.ln1(x)) # Normalize x. Run it through Attention, Add the result back to the original x. Why? This allows the original information to flow through the network without getting lost or "diluted." It's like having a copy of the original notes while you're adding new highlights.
            x = x + self.ff(self.ln2(x)) # Normalize again. Run it through the FeedForward "thinking" layers. Add it back again
            return x

print(text)
print(chars)
# print(stoi)
# print(itos)
# print(encode(stoi))
# print(decode(itos))
# print(data)
