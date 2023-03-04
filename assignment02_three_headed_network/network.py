
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.title = nn.Sequential(nn.Conv1d(hid_size, hid_size*2, kernel_size=3),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(output_size=4))
        
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.description = nn.Sequential(nn.Conv1d(hid_size, hid_size*2, kernel_size=3),
                                         nn.ReLU(), nn.AdaptiveAvgPool1d(output_size=4))
        # <YOUR CODE HERE>
        
        
        # <YOUR CODE HERE>
        self.category_out = nn.Sequential(nn.Linear(n_cat_features, hid_size),
                                        nn.ReLU(),
                                        nn.Linear(hid_size, hid_size*2))


        # Example for the final layers (after the concatenation)
#         self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
#         self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=concat_number_of_features, out_features=hid_size*4),
            nn.ReLU(),
            nn.Linear(in_features=hid_size*4, out_features=1))

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title(title_beg) # <YOUR CODE HERE>

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.description(full_beg) # <YOUR CODE HERE>        

        category = self.category_out(input3) # <YOUR CODE HERE>        

        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)

        out = self.classifier(concatenated) # <YOUR CODE HERE>

        return out