class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, output_size, hidden_size=128, num_layers = 1):

        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = 128
        self.num_layers = 1
        #self.embedding = nn.Linear(input_size, 64)
        self.lstm = nn.LSTM(134, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first=False)
        self.linear = nn.Linear(self.hidden_size, output_size)           

    def forward(self, encoder_hidden_cat, hidden):
        
        lstm_out, self.hidden = self.lstm(encoder_hidden_cat, hidden)   # use encoder's final hidden state concatted with one-hot as input
        output = self.linear(lstm_out)     
        return output, self.hidden


        