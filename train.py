import torchtext
import torch
from torch_struct import SentCFG
from torch_struct.networks import NeuralCFG
import torch_struct.data

# Download and the load default data.
WORD = torchtext.data.Field(include_lengths=True)
UD_TAG = torchtext.data.Field(
    init_token="<bos>", eos_token="<eos>", include_lengths=True
)

# Download and the load default data.
train, val, test = torchtext.datasets.UDPOS.splits(
    fields=(("word", WORD), ("udtag", UD_TAG), (None, None)),
    filter_pred=lambda ex: 5 < len(ex.word) < 30,
)

WORD.build_vocab(train.word, min_freq=3)
UD_TAG.build_vocab(train.udtag)
train_iter = torch_struct.data.TokenBucket(train, batch_size=200, device="cuda:0")

H = 256
T = 30
NT = 30
model = NeuralCFG(len(WORD.vocab), T, NT, H)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.75, 0.999])


def train():
    # model.train()
    losses = []
    for epoch in range(10):
        for i, ex in enumerate(train_iter):
            opt.zero_grad()
            words, lengths = ex.word
            N, batch = words.shape
            words = words.long()
            params = model(words.cuda().transpose(0, 1))
            dist = SentCFG(params, lengths=lengths)
            loss = dist.partition.mean()
            (-loss).backward()
            losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            if i % 100 == 1:
                print(-torch.tensor(losses).mean(), words.shape)
                losses = []


train()
