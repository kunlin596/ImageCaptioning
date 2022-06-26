import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] # remove the last layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           bias=True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device='cuda'),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device='cuda'))
    
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        
        captions = captions[:, :-1] # Remove <end>
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features shape: (batch_size, embed_size)
        self.hidden = self.init_hidden(batch_size) 
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions_length - 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption_length, embed_size)
        out, self.hidden = self.rnn(embeddings, self.hidden) # out shape : (batch_size, caption_length, hidden_size)
        return self.fc(out) # out shape : (batch_size, caption length, vocab_size)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for i in range(max_len):
            out, states = self.rnn(inputs, states)
            out = self.fc(out.squeeze(1))
           
            # Predict the next word
            # tensor.max returns the max value and also the max index, get the index here
            wordindex = out.max(1)[1] 
            caption.append(wordindex.item())
            
            # Break when we reach <end>
            if wordindex == 1:
                break
            
            # Construct the next input using the predicted one
            inputs = self.word_embeddings(wordindex).unsqueeze(1)
        return caption