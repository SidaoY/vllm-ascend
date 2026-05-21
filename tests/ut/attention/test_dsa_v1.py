import torch

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm_ascend.attention.dsa_v1 import (
    _build_dsa_slot_mapping,
    _build_dsa_slot_mapping_padding_mask,
    _mask_padding_slot_values_,
)


def test_build_dsa_slot_mapping_maps_padding_to_null_block():
    slot_mapping = torch.tensor([0, 1, 127, 128, 255, PAD_SLOT_ID], dtype=torch.int32)

    dsa_slot_mapping = _build_dsa_slot_mapping(slot_mapping, block_size=128)

    expected = torch.tensor(
        [
            [NULL_BLOCK_ID, 0],
            [NULL_BLOCK_ID, 1],
            [NULL_BLOCK_ID, 127],
            [1, 0],
            [1, 127],
            [NULL_BLOCK_ID, 0],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(dsa_slot_mapping, expected)


def test_mask_padding_slot_values_zeros_only_padding_rows():
    slot_mapping = torch.tensor(
        [
            PAD_SLOT_ID,
            0,
            128,
        ],
        dtype=torch.int32,
    )
    padding_mask = _build_dsa_slot_mapping_padding_mask(slot_mapping)
    values = torch.ones(3, 1, 4)

    masked_values = _mask_padding_slot_values_(padding_mask, values)

    assert masked_values is values
    assert torch.equal(
        values,
        torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0]],
            ]
        ),
    )
