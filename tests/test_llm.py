import json
from unittest.mock import MagicMock

import pytest

from llm.kv_cache import KVCache
from llm.quantization import Quantizer
from llm.server import JSONRPCHandler
from llm.transformer import ZGPULLM, LLMConfig
from nn.linear import Linear
from runtime.runtime import ZRuntime


def test_linear_layer():
    rt = ZRuntime()
    # 4x4 input and linear layer with 4x4 weights
    layer = Linear(in_features=4, out_features=4, bias=True, runtime=rt)
    layer.weight = [[1.0] * 4 for _ in range(4)]
    layer.bias = [0.5] * 4

    x = [[1.0, 2.0, 3.0, 4.0]]
    out = layer(x)
    assert len(out) == 1
    assert len(out[0]) == 4
    # (1+2+3+4) * 1.0 + 0.5 = 10.5
    for val in out[0]:
        assert pytest.approx(val, rel=1e-3) == 10.5


def test_transformer_forward():
    rt = ZRuntime()
    cfg = LLMConfig(vocab_size=32, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_seq_len=16)
    model = ZGPULLM(config=cfg, runtime=rt)

    tokens = [5, 12, 19]
    logits = model.forward(tokens)
    assert len(logits) == 3
    assert len(logits[0]) == cfg.vocab_size


def test_transformer_generate():
    rt = ZRuntime()
    cfg = LLMConfig(vocab_size=32, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_seq_len=16)
    model = ZGPULLM(config=cfg, runtime=rt)

    prompt = [3, 7]
    generated = model.generate(prompt, max_new_tokens=4)
    assert len(generated) == 6
    assert generated[:2] == prompt


def test_server_rpc_methods():
    handler = JSONRPCHandler.__new__(JSONRPCHandler)
    handler.headers = {"Content-Length": "0"}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Test health method
    req_body = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "system.health"}).encode("utf-8")
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = req_body
    handler.headers = {"Content-Length": str(len(req_body))}
    handler.do_POST()

    written_data = b"".join(call.args[0] for call in handler.wfile.write.call_args_list)
    res = json.loads(written_data.decode("utf-8"))
    assert res["id"] == 10
    assert res["result"]["healthy"] is True


def test_quantization_int8():
    data = [-1.0, -0.5, 0.0, 0.5, 1.0]
    q_tensor = Quantizer.quantize_int8_symmetric(data, shape=(5,))
    assert q_tensor.bits == 8
    assert len(q_tensor.q_data) == 5
    assert q_tensor.q_data[0] == -127
    assert q_tensor.q_data[-1] == 127

    deq = q_tensor.dequantize()
    for orig, d in zip(data, deq, strict=False):
        assert pytest.approx(orig, abs=0.02) == d


def test_quantization_int4_pack_unpack():
    data = [-1.0, 0.0, 1.0, -0.5]
    q_tensor = Quantizer.quantize_int4_symmetric(data, shape=(4,))
    assert q_tensor.bits == 4
    assert len(q_tensor.q_data) == 4

    packed = Quantizer.pack_int4(q_tensor.q_data)
    assert len(packed) == 2

    unpacked = Quantizer.unpack_int4(packed, target_len=4)
    assert unpacked == q_tensor.q_data


def test_kv_cache():
    cache = KVCache(max_seq_len=8, num_layers=1, d_model=4)
    assert cache.current_seq_len == 0

    k0 = [1.0, 2.0, 3.0, 4.0]
    v0 = [0.1, 0.2, 0.3, 0.4]
    all_k, _all_v = cache.update(0, k0, v0)
    assert len(all_k) == 1
    assert all_k[0] == k0
    cache.advance_step()
    assert cache.current_seq_len == 1

    k1 = [5.0, 6.0, 7.0, 8.0]
    v1 = [0.5, 0.6, 0.7, 0.8]
    all_k2, _all_v2 = cache.update(0, k1, v1)
    assert len(all_k2) == 2
    assert all_k2[1] == k1


def test_thai_multilingual_tokenizer():
    from llm.thai_tokenizer import ThaiMultilingualTokenizer

    tok = ThaiMultilingualTokenizer(vocab_size=256)
    text = "สวัสดี zGPU ภาษาไทย"
    ids = tok.encode(text)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded == text
