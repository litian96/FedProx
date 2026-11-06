import unittest

import torch
from torch import nn

from flearn.pytorch_fedprox import (
    FedProxOptimizer,
    dequantize_model_delta,
    dequantize_tensor,
    quantize_model_delta,
    quantize_tensor,
)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


class TestPyTorchFedProx(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_proximal_term_restricts_drift(self):
        # Model 1: Standard SGD (mu = 0)
        # Model 2: FedProx (mu = 1.0)
        m_sgd = LinearModel()
        m_prox = LinearModel()
        m_prox.load_state_dict(m_sgd.state_dict())

        w_star = {k: v.clone() for k, v in m_sgd.state_dict().items()}

        opt_sgd = FedProxOptimizer(m_sgd.parameters(), lr=0.1, mu=0.0)
        opt_prox = FedProxOptimizer(m_prox.parameters(), lr=0.1, mu=1.0)
        opt_prox.set_global_weights(w_star)

        inputs = torch.randn(8, 10)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        for _ in range(10):
            opt_sgd.zero_grad()
            loss1 = criterion(m_sgd(inputs), targets)
            loss1.backward()
            opt_sgd.step()

            opt_prox.zero_grad()
            loss2 = criterion(m_prox(inputs), targets)
            loss2.backward()
            opt_prox.step()

        # Compute drift from w*
        drift_sgd = torch.norm(m_sgd.fc.weight.data - w_star["fc.weight"]).item()
        drift_prox = torch.norm(m_prox.fc.weight.data - w_star["fc.weight"]).item()

        # FedProx with mu=1.0 must keep parameters strictly closer to w*
        self.assertLess(drift_prox, drift_sgd)

    def test_quantize_dequantize_delta_roundtrip(self):
        delta = torch.randn(16, 32, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            delta, nbits=4, block_size=32
        )
        recon = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=32
        )

        cos_sim = torch.cosine_similarity(
            delta.flatten(), recon.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.95)

    def test_model_delta_compression(self):
        local_model = LinearModel()
        global_model = LinearModel()

        quant_dict, _residuals, stats = quantize_model_delta(
            local_model.state_dict(), global_model.state_dict(), nbits=4, block_size=32
        )

        self.assertIn("compression_ratio", stats)
        self.assertGreaterEqual(stats["compression_ratio"], 4.0)
        self.assertGreaterEqual(stats["bandwidth_savings_pct"], 75.0)

        dequant_delta = dequantize_model_delta(quant_dict)
        self.assertIn("fc.weight", dequant_delta)
        self.assertEqual(dequant_delta["fc.weight"].shape, local_model.fc.weight.shape)

    def test_server_aggregation_with_quantized_deltas(self):
        global_model = LinearModel()
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

        deltas = []
        for client_id in range(3):
            client_model = LinearModel()
            quant_dict, _, _ = quantize_model_delta(
                client_model.state_dict(), global_state, nbits=4, block_size=32
            )
            dequant = dequantize_model_delta(quant_dict)
            deltas.append(dequant)

        # Server aggregation: w_new = w* + avg(deltas)
        avg_delta = torch.zeros_like(global_state["fc.weight"])
        for d in deltas:
            avg_delta += d["fc.weight"] / 3.0

        new_weight = global_state["fc.weight"] + avg_delta
        self.assertEqual(new_weight.shape, global_state["fc.weight"].shape)
        self.assertFalse(torch.isnan(new_weight).any())

    def test_empty_tensor(self):
        empty = torch.empty((0,), dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            empty, nbits=4, block_size=32
        )
        recon = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=32
        )
        self.assertEqual(recon.numel(), 0)


if __name__ == "__main__":
    unittest.main()
