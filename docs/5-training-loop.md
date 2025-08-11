# 5. Training Loop

## 5-1. Data Loader
Notes
* start_idxs should be between "1 ~ seq_len-ctx_len-1" (0~seq_len-ctx_len-2)
    * seq_len-ctx_len-1: last token should have next prediction value
* input_ids, target_ids are (batch_size, ctx_len)
    * from all possible start_idx sample batch_size idxs

Testcases
```
tests/test_data.py::test_get_batch PASSED
```

## 5-2. Checkpointing

Testcases
```
tests/test_serialization.py::test_checkpointing PASSED
```