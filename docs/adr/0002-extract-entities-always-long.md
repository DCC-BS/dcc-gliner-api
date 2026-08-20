# /extract_entities always chunk-scans the full document

`max_len` truncation on the entity endpoints silently dropped everything past the encoder window. We replaced that behavior: `/extract_entities` and `/batch_extract_entities` always run a chunk scan (hardcoded `chunk_size=384`, `chunk_overlap=64`, matching the upstream defaults), and `max_len` was removed from the entity request models. This is a deliberate semantic change for existing callers: latency grows with document length, but recall no longer depends on document prefix. Other endpoints keep `max_len`.
