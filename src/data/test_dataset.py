import pytest
import torch
from transformers import AutoTokenizer

import dataset
from train_data_generate import Notebook


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")

@pytest.fixture
def notebook():
    return Notebook(
            "123",
            [
                ("import abc\n from abc import apple\n apple_ = apple.Apple()"),
                ("apple = 1 + x"),
                ("bear = panda"),
                ],
            [
                ("this is a test_sample"),
                ("there isn't a test_sample"),
                ],
            )


@pytest.mark.parametrize(
        "tests",
        [
            ("this is a test_sample"),
            ("import abc\n from abc import apple\n apple_ = apple.Apple()")
            ],
        )
def test_texts_tokenize(tokenizer, tests):
    results = dataset.texts_tokenize(tokenizer, tests)
    assert len(tests) == len(results)
    assert isinstance(results[0]["input_ids"],torch.Tensor)


def test_prepare_input(tokenizer, notebook):
    results = dataset.prepare_input(tokenizer, tokenizer, notebook)
    assert isinstance(results["code_input_ids"], list)
    assert len(results["code_input_ids"]) == 3
    assert len(results["md_input_ids"]) == 2
