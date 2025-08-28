import pytest
from chunking.models import ChunkMetadata, Chunk, DocumentMetadata


def test_chunk_metadata_creation():
    metadata = ChunkMetadata(
        chunk_id="test-1",
        doc_id="doc-1", 
        chunk_type="text",
        word_count=150
    )
    assert metadata.chunk_id == "test-1"
    assert metadata.chunk_type == "text"
    assert metadata.word_count == 150


def test_chunk_creation():
    metadata = ChunkMetadata(
        chunk_id="test-1",
        doc_id="doc-1",
        chunk_type="text", 
        word_count=100
    )
    chunk = Chunk(metadata=metadata, content="Test content")
    assert chunk.content == "Test content"
    assert chunk.embedding is None