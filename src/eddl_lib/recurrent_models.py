# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import numpy as np
from text.metrics import compute_bleu_edll as compute_bleu

def _shared_top(visual_dim, semantic_dim, emb_size, init_v):
    # INPUT: visual features
    cnn_top_in = eddl.Input([visual_dim], name="in_visual_features")
    visual_features = eddl.RandomUniform(eddl.Dense(cnn_top_in, cnn_top_in.output.shape[1], name="visual_features"), -init_v, init_v)
    alpha_v = eddl.Softmax(eddl.Dense(eddl.Tanh(visual_features), visual_features.output.shape[1], name="dense_alpha_v"), name="alpha_v")  # missing sentence component
    v_att = eddl.Mult(alpha_v, visual_features)
    print(f"layer visual features: {visual_features.output.shape}")

    # INPUT: semantic features
    cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
    semantic_features = eddl.RandomUniform(eddl.Embedding(
        eddl.ReduceArgMax(cnn_out_in, [0]), cnn_out_in.output.shape[1], 1, emb_size, name="semantic_features"), -init_v, init_v)
    alpha_s = eddl.Softmax(eddl.Dense(eddl.Tanh(semantic_features), emb_size, name="dense_alpha_s"), name="alpha_s")  # missing sentence component cnn_out.output.shape[1]
    s_att = eddl.Mult(alpha_s, semantic_features)
    print(f"layer semantic features: {semantic_features.output.shape}")

    # co-attention
    features = eddl.Concat([v_att, s_att], name="co_attention")  # name="co_att_in")
    #context = eddl.RandomUniform(eddl.Dense(features, emb_size, name="co_attention"), -2*init_v, 2*init_v)
    #print(f"layer coattention: {context.output.shape}")
    return cnn_top_in, cnn_out_in, features


def recurrent_lstm_model(visual_dim, semantic_dim, vs, emb_size, lstm_size, init_v=0.05):
    assert init_v > 0
    cnn_top_in, cnn_out_in, context = _shared_top(visual_dim, semantic_dim, emb_size, init_v)
    # lstm
    word_in = eddl.Input([vs])
    to_lstm = eddl.ReduceArgMax(word_in, [0])
    to_lstm = eddl.RandomUniform(eddl.Embedding(to_lstm, vs, 1, emb_size, mask_zeros=True, name="word_embs"), -init_v, init_v)
    to_lstm = eddl.Concat([to_lstm, context])
    lstm = eddl.LSTM(to_lstm, lstm_size, mask_zeros=True, bidirectional=False, name="lstm_cell")
    eddl.setDecoder(word_in)
    out_lstm = eddl.Softmax(eddl.Dense(lstm, vs, name="out_dense"), name="rnn_out")
    print(f"layer lstm, output shape: {out_lstm.output.shape}")
    # model
    rnn = eddl.Model([cnn_top_in, cnn_out_in], [out_lstm])
    return rnn


# model used for generating text (test-inference stage)
def nonrecurrent_lstm_model(visual_dim, semantic_dim, vs, emb_size, lstm_size, init_v=0.05):
    cnn_top_in, cnn_out_in, context = _shared_top(visual_dim, semantic_dim, emb_size, init_v)
    
    lstm_in = eddl.Input([vs])
    lstate = eddl.States([2, lstm_size])
    
    to_lstm = eddl.ReduceArgMax(lstm_in, [0])  # word index
    to_lstm = eddl.Embedding(to_lstm, vs, 1, emb_size, name="word_embs")
    to_lstm = eddl.Concat([to_lstm, context])
    lstm = eddl.LSTM([to_lstm, lstate], lstm_size, True, name="lstm_cell")
    lstm.isrecurrent = False
    
    out_lstm = eddl.Softmax(
                eddl.Dense(lstm, vs, name="out_dense"), 
                name="rnn_out")
    
    # *** model
    model = eddl.Model([cnn_top_in, cnn_out_in, lstm_in, lstate], [out_lstm])
    # if the model is saved in onnx, there is the same error as in the recurrent model when loaded: 
    #       LDense only works over 2D tensors (LDense)
    return model


def generate_text(rnn, n_tokens, visual_batch=None, semantic_batch=None, dev=False):
    assert (visual_batch is not None) and (semantic_batch is not None)

    bs = visual_batch.shape[0]
    lstm = eddl.getLayer(rnn, "lstm_cell")
    lstm_size = lstm.output.shape[1]
    last_layer = eddl.getLayer(rnn, "rnn_out")
    voc_size = last_layer.output.shape[1]
    
    # return value
    generated_tokens = np.zeros( (bs, n_tokens), dtype=int)

    # lstm cell states
    state_t = Tensor.zeros([bs, 2, lstm_size])
    
    # token: input to lstm cell
    token = Tensor.zeros([bs, voc_size])
    
    for j in range(0, n_tokens):
        if dev:
            print(f" *** token {j}/{n_tokens} ***")
            print(f"cnn_visual: {visual_batch.shape}")
            print(f"cnn_semant: {semantic_batch.shape}")
            print(f"token: {token.shape}")
            print(f"state_t: {state_t.shape}")

        # forward: token and state_t update after the forward step
        eddl.forward(rnn, [visual_batch, semantic_batch, token, state_t])     
        states = eddl.getStates(lstm)

        # save the state for the next token: it must be copied into a Tensor (state_t)
        for si in range(len(states)):
            states[si].reshape_([ states[si].shape[0], 1, states[si].shape[1] ])
            state_t.set_select( [":", str(si), ":"] , states[si] )
        
        out_soft = eddl.getOutput(last_layer)
        # pass control to numpy for argmax
        wis = np.argmax(out_soft, axis=-1)
        # if dev:
        #     print(wis.shape)
        #     print(f"next_token {wis[0]}")
        generated_tokens[:, j] = wis
        
        #> next input token to the lstm
        word_index = Tensor.fromarray(wis.astype(float))
        word_index.reshape_([bs, 1])  # add dimension for one-hot encoding
        token = Tensor.onehot(word_index, voc_size)
        # print(token.shape)
        token.reshape_([bs, voc_size])  # remove singleton dim
        #<
    #< for n_tokens
    return generated_tokens
#<