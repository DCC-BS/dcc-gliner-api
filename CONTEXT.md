# DCC GLiNER API

HTTP API around the GLiNER2 model for schema-based information extraction, serving DCC-BS document pipelines.

## Language

**Deployment**:
The Ray Serve + FastAPI layer that translates HTTP requests into service calls and shapes responses.
_Avoid_: Controller, handler, server

**Service**:
The wrapper that owns the GLiNER2 model lifecycle and every model interaction. The only code that touches the model.
_Avoid_: Engine, repository, client

**Chunking**:
The pure, model-free module that splits long documents into overlapping chunks and merges per-chunk detections.
_Avoid_: Splitter, segmentation, windowing

**Chunk scan**:
Processing a document by extracting from each chunk in sequence and merging the detections, instead of truncating to the encoder window.
_Avoid_: Long read, pagination

**Chunk**:
One overlapping word window of a document, with offsets into the original text.
_Avoid_: Window, slice, segment

**Global span**:
A `(start, end)` character offset pair that indexes the original document, not a chunk.
_Avoid_: Position, offset (ambiguous)

**Mention**:
One detected occurrence of an entity label at a specific global span. Distinct mentions at different positions are never collapsed.
_Avoid_: Entity (ambiguous — that's the label), match

**Overlap artifact**:
The same mention detected by two adjacent chunks because they share words. Removed during merge.
_Avoid_: Duplicate (too broad — distinct repeated mentions are not artifacts)

**Label**:
An entity type requested by the caller, optionally with a description.
_Avoid_: Class, category, tag
