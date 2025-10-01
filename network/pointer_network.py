import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_cuda=False):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # self.hidden_0 = self.get_hidden_0(hidden_dim)
        # self.h_0 = torch.zeros(1, requires_grad=False)
        # self.c_0 = torch.zeros(1, requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        # 变为length*batch_size*embd_size
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        output, hidden = self.lstm(embedded_inputs, hidden)
        output = output.permute(1, 0, 2)
        return output, hidden

    def get_hidden_0(self, embedded_inputs):
        batch_size = embedded_inputs.size(0)
        h_0 = torch.zeros([1, batch_size, self.hidden_dim], requires_grad=False)
        c_0 = torch.zeros([1, batch_size, self.hidden_dim], requires_grad=False)
        if self.use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        # h_0 = self.h_0.unsqueeze(0).unsqueeze(0).repeat(1, batch_size, self.hidden_dim)
        # c_0 = self.c_0.unsqueeze(0).unsqueeze(0).repeat(1, batch_size, self.hidden_dim)
        return h_0, c_0


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_cuda=False):
        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)

        self.V = nn.Parameter(torch.zeros(hidden_dim, requires_grad=True))
        nn.init.uniform_(self.V, -1, 1)

        self._inf = nn.Parameter(
            torch.FloatTensor([float("-inf")]), requires_grad=False
        )

        # if use_cuda:
        #     self.V=self.V.cuda()
        #     # self._inf=self._inf.cuda()

        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax()

    def forward(self, x, contex, mask):
        inp = self.input_linear(x).unsqueeze(2).expand(-1, -1, contex.size(1))
        contex = contex.permute(0, 2, 1)
        ctx = self.context_linear(contex)
        V = self.V.unsqueeze(0).expand(contex.size(0), -1).unsqueeze(1)

        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        # alpha = self.softmax(att,dim=1)
        alpha = torch.softmax(att, dim=1)
        # alpha=torch.log_softmax(att,dim=1)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_cuda=False):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hiddden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(2 * hidden_dim, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim, use_cuda=use_cuda)

        self.mask = nn.Parameter(
            torch.ones(1, requires_grad=False), requires_grad=False
        )
        self.runner = nn.Parameter(
            torch.zeros(1, requires_grad=False), requires_grad=False
        )
        # if use_cuda:
        #     self.mask=self.mask.cuda()
        #     self.runner=self.runner.cuda()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, embedded_inputs, decoder_input, hidden, context):
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # runner = torch.arange(input_length, requires_grad=False)
        # if self.use_cuda:
        #     runner=runner.cuda()
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hiddden_to_hidden(h)
            input_, forget, cell, out = gates.chunk(4, 1)
            input_ = self.sigmoid(input_)
            forget = self.sigmoid(forget)
            cell = self.tanh(cell)
            out = self.sigmoid(out)

            c_t = (forget * c) + (input_ * cell)
            h_t = self.tanh(c_t) * out

            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = self.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            masked_outs = outs * mask

            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (
                runner == indices.unsqueeze(1).expand(-1, outs.size()[1])
            ).float()

            mask = mask * (1 - one_hot_pointers)

            embedding_mask = (
                one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            )
            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.embedding_dim
            )

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_cuda=False):
        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hideen_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embedding = nn.Linear(1, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim, use_cuda=use_cuda)
        self.decoder = Decoder(embedding_dim, hidden_dim, use_cuda=use_cuda)
        self.decoder_input_0 = nn.Parameter(
            torch.zeros(embedding_dim, requires_grad=False), requires_grad=False
        )
        # if use_cuda:
        #     self.decoder_input_0=self.decoder_input_0.cuda()
        nn.init.uniform_(self.decoder_input_0, -1, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input_0 = self.decoder_input_0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden_0 = self.encoder.get_hidden_0(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(
            embedded_inputs, encoder_hidden_0
        )

        decoder_hidden_0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs, decoder_input_0, decoder_hidden_0, encoder_outputs
        )

        return outputs, pointers


def get_pointer_network(embedding_dim, hidden_dim, use_cuda=False):
    pointer_network = PointerNet(embedding_dim, hidden_dim, use_cuda=use_cuda)
    if use_cuda:
        pointer_network = pointer_network.cuda()
    return pointer_network


# def get_reward():
#     return 1

# pointer_network=PointerNet(5,6)
# x=torch.rand(3,4)
# out=pointer_network(x)
# print(f'out={out}')
# pointer_network=PointerNet(5,6,use_cuda=True).cuda()
# x=torch.rand(3,4).cuda()
# out=pointer_network(x)
# print(f'out={out}')
