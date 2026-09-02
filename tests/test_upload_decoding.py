"""Decoding of an uploaded document body."""

import gzip

import pytest
from fastapi import HTTPException

from dcc_gliner_api.app import _decode_upload


def test_plain_utf8_bytes_read_back():
    assert _decode_upload("Grüezi Basel".encode(), "utf-8") == "Grüezi Basel"


def test_gzipped_bytes_are_unpacked():
    raw = gzip.compress("Grüezi Basel".encode())
    assert _decode_upload(raw, "utf-8") == "Grüezi Basel"


def test_other_charset_is_honoured():
    assert _decode_upload("Grüezi".encode("latin-1"), "latin-1") == "Grüezi"


def test_undecodable_bytes_are_a_bad_request():
    with pytest.raises(HTTPException) as excinfo:
        _decode_upload(b"\xff\xfe\xfa", "utf-8")
    assert excinfo.value.status_code == 400
