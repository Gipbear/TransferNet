import unittest

import torch

from CompWebQ.model import propagate_triples


def _loop_propagate(last_e, rel_dist, triples, triple_batch, num_ents):
    outputs = []
    for batch_idx in range(last_e.shape[0]):
        sample_triples = triples[triple_batch == batch_idx]
        sub, rel, obj = sample_triples[:, 0], sample_triples[:, 1], sample_triples[:, 2]
        contributions = last_e[batch_idx:batch_idx + 1, sub] * rel_dist[batch_idx:batch_idx + 1, rel]
        outputs.append(torch.index_add(last_e.new_zeros(1, num_ents), 1, obj, contributions))
    return torch.cat(outputs, dim=0)


class TestCompWebQVectorizedPropagation(unittest.TestCase):
    def test_matches_loop_with_duplicate_object_entities(self):
        triples = torch.tensor([
            [0, 0, 2], [1, 1, 2], [2, 0, 3], [0, 1, 1], [1, 0, 1],
        ])
        triple_batch = torch.tensor([0, 0, 0, 1, 1])
        last_e = torch.tensor([[0.2, 0.4, 0.8, 0.1], [0.7, 0.3, 0.5, 0.2]])
        rel_dist = torch.tensor([[0.5, 0.25], [0.4, 0.6]])

        expected = _loop_propagate(last_e, rel_dist, triples, triple_batch, num_ents=4)
        actual = propagate_triples(last_e, rel_dist, triples, triple_batch, num_ents=4)

        torch.testing.assert_close(actual, expected)

    def test_preserves_gradients(self):
        triples = torch.tensor([[0, 0, 2], [1, 1, 2], [0, 1, 1]])
        triple_batch = torch.tensor([0, 0, 1])
        last_e = torch.tensor([[0.2, 0.4, 0.8], [0.7, 0.3, 0.5]], requires_grad=True)
        rel_dist = torch.tensor([[0.5, 0.25], [0.4, 0.6]], requires_grad=True)
        actual = propagate_triples(last_e, rel_dist, triples, triple_batch, num_ents=3)
        actual.square().sum().backward()
        actual_grads = (last_e.grad.clone(), rel_dist.grad.clone())

        last_e_ref = last_e.detach().clone().requires_grad_()
        rel_dist_ref = rel_dist.detach().clone().requires_grad_()
        expected = _loop_propagate(last_e_ref, rel_dist_ref, triples, triple_batch, num_ents=3)
        expected.square().sum().backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], last_e_ref.grad)
        torch.testing.assert_close(actual_grads[1], rel_dist_ref.grad)


if __name__ == "__main__":
    unittest.main()
