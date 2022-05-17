# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import pickle
from posixpath import join

import torch
import torchvision.models as models
import torch.onnx
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# global initialization functions
def _init_linear(t, v):
    v = abs(v)
    torch.nn.init.uniform_(t, -v, v)

def _init_embs(t, v):
    _init_linear(t, v)

# --------------------------------------------------
class CNN(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.conv_net = dict()
        self.conv_layers, self.n_out_features, self.avg_func, self.out = self._create_net()

    def _create_net(self):
        resnet = models.resnet18(pretrained=True)
        # print(resnet)
        inner_modules = list(resnet.children())[:-2]
        if self.conf["single_channel_cnn"]:
            print("*** replacing input channel of the pretrained cnn, color channels from 3 to 1")
            # replace input layer to accept 
            fl = inner_modules[0]
            fl2 = torch.nn.Conv2d(in_channels=1, out_channels=fl.out_channels,
                    kernel_size = fl.kernel_size, stride=fl.stride, padding=fl.padding, bias=fl.bias)
            fl2.weight.data[:, 0, :] = fl.weight[:, 0, :]
            inner_modules[0] = fl2
        #< single_channel_cnn
        n_out_features = resnet.fc.in_features
        cnn = nn.Sequential(*inner_modules)  # output is 512x7x7
        # freeze training
        for param in cnn.parameters():
            param.requires_grad = False

        # avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        avg_func = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        out = nn.Linear(in_features=n_out_features, out_features=n_out_features)
        _init_linear(out.weight, self.conf["init_linear"])
        return cnn, n_out_features, avg_func, out

    def forward(self, images):        
        conv_features = self.conv_layers(images)
        avg_features = self.avg_func(conv_features)  # [bs, 512, 1, 1] -- not using squeeze for preserving bs when bs==1
        # print("before reshape:", avg_features.shape)
        avg_features = avg_features.reshape(avg_features.shape[0], avg_features.shape[1])
        # print("after reshape:", avg_features.shape)
        est = self.out(avg_features)
        return conv_features, avg_features, est

    #>>> 
    # useless now, kept here if the cnn needs to be detached from the model and trained on its own (as in EDDL pipeline)
    # beware: this is old code, in the current implementation it might not work in its current form
    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.conf["learning_rate"], momentum=self.conf["momentum"])
    #     lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=5, cooldown=0, verbose=True)
    #     return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'avg_val_loss'}

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     _, _, y_hat = self.forward(x)
    #     loss =  F.binary_cross_entropy_with_logits(y_hat, torch.squeeze(y))  # F.cross_entropy(y_hat, y)
    #     return {'loss':  loss}
    #<<<

#< base model: cnn

# --------------------------------------------------
class MultiLabelTags(nn.Module):
    def __init__(self, n_in_features, conf):
        super().__init__()
        self.n_in_features = n_in_features
        self.conf = conf
        self.k_tags = self.conf["top_k_tags"]
        self.linear, self.embeddings =  self._create_net()

    def _create_net(self):
        linear = nn.Linear(in_features=self.n_in_features, out_features=self.conf["n_tags"])
        _init_linear(linear.weight, self.conf["init_linear"])
        
        embeddings = nn.Embedding(self.conf["n_tags"], self.conf["tag_emb_size"])
        _init_embs(embeddings.weight, self.conf["init_embs"])  
        return linear, embeddings

    def forward(self, features):
        tags = F.softmax(self.linear(features), dim=1)  # SOFTWMAX WITH L1 NORMALIZED, ELSE: SIGMOID
        embs = self.embeddings(torch.topk(tags, self.k_tags)[1])  # torch.topk returns (values, indices)
        return tags, embs
#< MultiLabelTags

# --------------------------------------------------
class CoAttention(nn.Module):
    def __init__(self, visual_feat_size, conf):
        super().__init__()
        self.visual_feat_size = visual_feat_size
        self.conf = conf
        self.lstm_size = self.conf["lstm_size"]
        self.DEBUG = self.conf["debug"]
        
        # VISUAL
        il = self.conf["init_linear"]
        self.visual_W_features = nn.Linear(in_features=self.visual_feat_size, out_features=self.visual_feat_size)
        _init_linear(self.visual_W_features.weight, il)
        self.visual_W_lstm = nn.Linear(in_features=self.lstm_size, out_features=self.visual_feat_size)
        _init_linear(self.visual_W_lstm.weight, il)
        self.visual_W = nn.Linear(in_features=self.visual_feat_size, out_features=self.visual_feat_size)
        _init_linear(self.visual_W.weight, il)

        # SEMANTIC
        self.seman_W_tags = nn.Linear(in_features=self.conf["tag_emb_size"], out_features=self.conf["lstm_size"])
        _init_linear(self.seman_W_tags.weight, il)
        self.seman_W_lstm = nn.Linear(in_features=self.conf["lstm_size"], out_features=self.conf["lstm_size"])
        _init_linear(self.seman_W_lstm.weight, il)
        self.seman_W = nn.Linear(in_features=self.conf["lstm_size"], out_features=self.conf["lstm_size"])
        _init_linear(self.seman_W.weight, il)

        # CO-ATT
        self.coatt_W = nn.Linear(in_features = self.visual_feat_size + self.conf["lstm_size"], out_features = self.conf["attn_emb_size"])
        _init_linear(self.coatt_W.weight, il)
    #< init

    def forward(self, visual_features, semantic_features, lstm_hidden_state):
        # visual
        vis_feat = self.visual_W_features(visual_features)
        # debug message left here commented
        #print("vis_feat:", vis_feat.shape)
        #print(f"lstm_hidden_state: {lstm_hidden_state.shape}")
        #print(next(self.visual_W_features.parameters()).is_cuda)
        #print(lstm_hidden_state.is_cuda)

        vis_lstm = self.visual_W_features(lstm_hidden_state.squeeze(1))
        #print("vis_lstm:", vis_lstm.shape)
        visual = self.visual_W(torch.tanh(torch.add(vis_feat, vis_lstm)))
        # print("visual", visual.shape)
        alpha_visual = F.softmax(visual, dim=1)
        visual_attention = torch.mul(alpha_visual, visual_features)
        # print("visual_attention:", visual_attention.shape)

        sem_tags = self.seman_W_tags(semantic_features)
        # print("sem_tags.shape:", sem_tags.shape)
        sem_lstm = self.seman_W_lstm(lstm_hidden_state)
        # print("sem_lstm.shape:", sem_lstm.shape)
        semantic = self.seman_W(torch.tanh(torch.add(sem_tags, sem_lstm)))
        # print("semantic:", semantic.shape)
        alpha_semantic = F.softmax(semantic, dim=1)
        semantic_attention = torch.mul(alpha_semantic, semantic_features)
        semantic_attention = torch.sum(semantic_attention, dim=1)
        # print("semantic_attention:", semantic_attention.shape)
        context = self.coatt_W(torch.cat([visual_attention, semantic_attention], dim=1))
        
        return context, alpha_visual, alpha_semantic
#< CoAttention

# --------------------------------------------------
class SentenceModule(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.lstm, self.W_topic_lstm, self.W_topic_context, self.W_stop_s_1, self.W_stop_s, self.W_stop = self._create_net()
    
    def _create_net(self):
        lstm = nn.LSTM(input_size=self.conf["attn_emb_size"],
                        hidden_size=self.conf["lstm_sent_h_size"],
                        num_layers=self.conf["lstm_sent_n_layers"],
                        dropout= self.conf["lstm_sent_dropout"] if self.conf["lstm_sent_n_layers"] < 1 else 0,
                        batch_first = True
                        )

        il = self.conf["init_linear"]
        # topic for word lstm
        W_topic_lstm = nn.Linear(in_features=self.conf["lstm_sent_h_size"], out_features=self.conf["attn_emb_size"])
        _init_linear(W_topic_lstm.weight, il)
        W_topic_context = nn.Linear(in_features=self.conf["attn_emb_size"], out_features=self.conf["attn_emb_size"])
        _init_linear(W_topic_context.weight, il)

        # stop probabilities
        W_stop_s_1 = nn.Linear(in_features=self.conf["lstm_sent_h_size"], out_features=self.conf["attn_emb_size"])
        _init_linear(W_stop_s_1.weight, il)
        W_stop_s = nn.Linear(in_features=self.conf["lstm_sent_h_size"], out_features=self.conf["attn_emb_size"])
        _init_linear(W_stop_s.weight, il)
        W_stop = nn.Linear(in_features=self.conf["attn_emb_size"], out_features=2)
        _init_linear(W_stop.weight, il)

        return lstm, W_topic_lstm, W_topic_context, W_stop_s_1, W_stop_s, W_stop

    def forward(self, context, prev_hidden_state, states=None):
        context = context.unsqueeze(1)  # add second dimension
        hidden_state, states = self.lstm(context, states)
        topic = torch.tanh(self.W_topic_lstm(hidden_state) + self.W_topic_context(context))
        topic = topic.squeeze(1)  # singleton dimension at pos 1
        p_stop = self.W_stop(torch.tanh(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))
        
        return hidden_state, states, topic, p_stop
#< SentenceModule

# --------------------------------------------------
class WordModule(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.embeddings, self.lstm, self.linear = self._create_net()
    
    def _create_net(self):
        with open( join(self.conf["exp_fld"], "vocab.pkl"), "rb") as fin:
            vocab = pickle.load(fin)
        voc_size = len(vocab.word2idx)
        embeddings = nn.Embedding(voc_size, self.conf["word_emb_size"], padding_idx=0)
        _init_embs(embeddings.weight, self.conf["init_embs"])
        lstm = nn.LSTM(self.conf["attn_emb_size"], self.conf["lstm_word_h_size"], self.conf["lstm_word_n_layers"], 
                    dropout= self.conf["lstm_word_dropout"] if self.conf["lstm_word_n_layers"] < 1 else 0, batch_first=True)
        linear = nn.Linear(self.conf["lstm_word_h_size"], voc_size)
        _init_linear(linear.weight, self.conf["init_linear"])
        return embeddings, lstm, linear

    
    def forward(self, topic, tokens):
        import numpy
        # print("word.forward, topic:", topic.shape)
        # print("word.forward, tokens:", tokens.shape)
        word_embs = self.embeddings(tokens)  # .squeeze(1)
        # print("word.forward, word_embs:", word_embs.shape)
        to_lstm = torch.cat( (topic.unsqueeze(1), word_embs), 1)

        # print("word.forward, to_lstm:", to_lstm.shape)
        hidden, _ = self.lstm(to_lstm)
        # print("word lstm hidden:", hidden.shape)
        output = self.linear(hidden[:, -1, :])
        return output
#< WordModule   


# --------------------------------------------------
# TEST
if __name__ == "__main__":
    bs = 4
    conf = {}
    conf["top_k_tags"] = 7
    conf["n_tags"] = 10
    conf["tag_emb_size"] = 512
    conf["lstm_size"] = 512
    conf["attn_emb_size"] = 512
    conf["lstm_sent_h_size"] = 512
    conf["lstm_sent_n_layers"] = 1
    conf["lstm_sent_dropout"] = 0
    conf["lstm_word_h_size"] = 512
    conf["lstm_word_n_layers"] = 1
    conf["lstm_word_dropout"] = 0
    conf["voc_size"] = 1000
    conf["word_emb_size"] = 512
    conf["init_linear"] = 0.1
    conf["init_embs"] = 1

    # INPUT DATA
    print("\nINPUT DATA")
    batch_img = torch.randn((bs, 3, 224, 224))
    batch_tokens = torch.randint(0, conf["voc_size"], (bs, 5))
    print("batch_img:", batch_img.shape)
    print("batch_tokens:", batch_tokens.shape)
    print(20 * "-")

    print("\nCONV")
    cnn = CNN(conf)
    conv, avg_conv, y_est = cnn.forward(batch_img)
    print("conv features:", conv.shape)
    print("avg_conv:", avg_conv.shape)
    print("y_est:", y_est.shape)
    print("---")

    print("MULTILABEL")
    print("- receiving input from cnn, size:", cnn.n_out_features)
    mltags = MultiLabelTags(cnn.n_out_features, conf)
    tags, embs = mltags.forward(avg_conv)
    print("(out) tags:", tags.shape)
    print("(out) tag embeddings embs.shape:", embs.shape)
    # print("tags (value):", tags)
    print("---")
    
    print("COATTENTION")
    coatt = CoAttention(512, conf)  # visual_feat_size 512
    lstm_state = torch.zeros((bs, 1, 512))
    context, a_v, a_s = coatt.forward(avg_conv, embs, lstm_state)
    print("(out) context:", context.shape)
    print("(out) a_v:", a_v.shape)
    print("(out) a_s:", a_s.shape)

    print("SENTENCE LSTM")
    sentenceLSTM = SentenceModule(conf)
    hidden_state, states, topic, prob_stop = sentenceLSTM.forward(context, lstm_state)
    print("(out) hidden_state", hidden_state.shape)
    #print(states.shape)
    print("(out) topic", topic.shape)
    print("(out) prob_stop", prob_stop.shape)

    print("WORD LSTM")
    wordLSTM = WordModule(conf)
    text = wordLSTM.forward(topic, batch_tokens)
    print("predicted text:", text.shape)