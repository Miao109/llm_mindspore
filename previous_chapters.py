import mindspore.nn as nn
import mindspore
import mindspore.ops as ops
from transformers import GPT2Tokenizer
from mindspore.dataset import GeneratorDataset
import mindspore.common.dtype as mstype
import logging
# 定义检查NaN值的函数
def check_nan(tensor):
    # 使用isnan函数检查每个元素是否为NaN
    isnan_tensor = ops.isnan(tensor)
    # 使用any函数判断是否存在NaN值
    has_nan = ops.any(isnan_tensor)
    return has_nan.asnumpy()

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-tokenizer")
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = GeneratorDataset(dataset, ["input_ids", "target_ids"], shuffle=shuffle)
    dataloader = dataloader.batch(batch_size, drop_remainder=drop_last)
    return dataloader

class GPTDatasetV1:
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(Tensor(input_chunk))
            self.target_ids.append(Tensor(target_chunk))
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    def __len__(self):
        return len(self.input_ids)
    
class MultiHeadAttention(nn.Cell):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Dense(d_in, d_out, has_bias=qkv_bias,)
        self.W_key = nn.Dense(d_in, d_out, has_bias=qkv_bias,)
        self.W_value = nn.Dense(d_in, d_out, has_bias=qkv_bias,)
        self.out_proj = nn.Dense(d_out, d_out,)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(p=dropout,)
        # 创建掩码矩阵
        ones_matrix = ops.ones((context_length, context_length), mindspore.float32)
        mask = ops.triu(ones_matrix, diagonal=1)
        self.mask = mindspore.Parameter(mask, requires_grad=False)

    def construct(self, x):
        b, num_tokens, d_in = x.shape
        # logging.warning("attn_scores：%s"%(x.dtype))
        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑张量以分离头
        keys = ops.reshape(keys, (b, num_tokens, self.num_heads, self.head_dim))
        values = ops.reshape(values, (b, num_tokens, self.num_heads, self.head_dim))
        queries = ops.reshape(queries, (b, num_tokens, self.num_heads, self.head_dim))

        # 交换维度
        keys = ops.transpose(keys, (0, 2, 1, 3))
        queries = ops.transpose(queries, (0, 2, 1, 3))
        values = ops.transpose(values, (0, 2, 1, 3))

        # 计算注意力分数
        keys_transposed = ops.transpose(keys, (0, 1, 3, 2))
        # logging.warning("attn_scores：%s, %s"%(queries.dtype, keys_transposed.dtype))
        attn_scores = ops.matmul(queries, keys_transposed)

        # 准备掩码
        mask_bool = self.mask[:num_tokens, :num_tokens].astype(mindspore.bool_)
        neg_inf = mindspore.Tensor(-float('inf'))
        attn_scores = ops.masked_fill(attn_scores, mask_bool, neg_inf)

        # 计算注意力权重
        scale_factor = keys.shape[-1] ** 0.5
        scaled_attn_scores = attn_scores / scale_factor
        # print("scaled_attn_scores3:", check_nan(scaled_attn_scores))
        softmax = nn.Softmax(axis=-1, )
        attn_weights = softmax(scaled_attn_scores)
        # print("attn_weights4:", check_nan(attn_weights))
        # if  check_nan(attn_weights):
        #     # 找出NaN值的索引
        #     nan_indices = ops.nonzero(ops.isnan(attn_weights)).squeeze()
        #     nan_indices = nan_indices.asnumpy().tolist()
        #     for ind in nan_indices:
        #         print(attn_weights[ind[0], ind[1], ind[2], ind[3]])
        #         print(scaled_attn_scores[ind[0], ind[1], ind[2], ind[3]])

        attn_weights = self.dropout(attn_weights)
        # print("attn_weights5:", check_nan(attn_weights))
        # 计算上下文向量
        context_vec = ops.matmul(attn_weights, values)
        # print("context_vec:", check_nan(context_vec))
        context_vec = ops.transpose(context_vec, (0, 2, 1, 3))
        
        # 合并头
        context_vec = ops.reshape(context_vec, (b, num_tokens, self.d_out))
        context_vec = self.out_proj(context_vec)

        return context_vec


# from chapter03 import MultiHeadAttention
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor
import mindspore

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        # with mindspore.context.grad_off():
        logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        softmax = ops.Softmax(axis=-1)
        probas = softmax(logits)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = ops.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = ops.concat((idx, idx_next), axis=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = Tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
     # 检查是否有批量维度，如果有则移除
    if len(token_ids.shape) > 1 and token_ids.shape[0] == 1:
        flat = token_ids.squeeze(0)
    else:
        flat = token_ids
    # flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat)


class LayerNorm(nn.Cell):
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = mindspore.Parameter(ops.ones(emb_dim, dtype=mindspore.float32))
        self.shift = mindspore.Parameter(ops.zeros(emb_dim, dtype=mindspore.float32))

    def construct(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / np.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Cell):
    def __init__(self):
        super(GELU, self).__init__()


    def construct(self, x):
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(Tensor(2.0 / np.pi,)) * (x + 0.044715 * ops.pow(x, 3))
        ))
    
class FeedForward(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.SequentialCell(
            nn.Dense(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Dense(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def construct(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Cell):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(p=cfg["drop_rate"])

    def construct(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

    
class GPTModel(nn.Cell):
    def __init__(self, cfg):
        super(GPTModel, self).__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(p=cfg["drop_rate"], dtype=mstype.float64)

        self.trf_blocks = nn.SequentialCell(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Dense(
            cfg["emb_dim"], cfg["vocab_size"], has_bias=False
        )

    def construct(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # print("seq_len:", seq_len, in_idx)
        tok_embeds = self.tok_emb(in_idx)
        # print("tok_embeds:", tok_embeds)
        pos_indices = Tensor(np.arange(seq_len), dtype=mindspore.int32)
        # print("pos_indices:", pos_indices)
        pos_embeds = self.pos_emb(pos_indices)
        # print("pos_embeds:", pos_embeds)
        
        pos_embeds = mindspore.ops.expand_dims(pos_embeds, 0)
        # print("pos_embeds:", pos_embeds)
        pos_embeds = mindspore.ops.tile(pos_embeds, (batch_size, 1, 1))
        # print("pos_embeds:", pos_embeds)

        x = tok_embeds + pos_embeds
        # print("x1:", x)

        x = self.drop_emb(x)
        # logging.warning("x2:%s"%(x.dtype))
        x = self.trf_blocks(x)
        # print("x3:", x)
        x = self.final_norm(x)
        # print("x4:", x)
        logits = self.out_head(x)
        # print("logits:", logits)
        return logits
    
import numpy
from mindspore import Tensor
def assign(left, right):
    # print(f"Shape. Left: {left.shape}, Right: {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return mindspore.Parameter(Tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.embedding_table = assign(gpt.pos_emb.embedding_table, params['wpe'])
    gpt.tok_emb.embedding_table = assign(gpt.tok_emb.embedding_table, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = numpy.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = numpy.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx = ops.cast(idx, mindspore.int32)
        idx_cond = idx[:, -context_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = ops.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = ops.where(logits < min_val, Tensor(float("-inf")), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            softmax = nn.Softmax(axis=-1)
            probs = softmax(logits)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = ops.multinomial(probs, num_samples=1, replacement=False)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = ops.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        # print(idx_next.dtype, idx.dtype)
        if idx_next.squeeze() == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = ops.cat((idx.astype(mindspore.int64), idx_next), axis=1)  # (batch_size, num_tokens+1)

    return idx

def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)
    loss = ops.cross_entropy(ops.flatten(logits, start_dim=0, end_dim=1), target_batch.flatten())
    print("calc_loss_batch:", loss)
    return loss


def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
            print(loss, total_loss)
        else:
            break
    return total_loss / num_batches