import base64
import json
import zlib

import pytest

from factorio_patches.blueprint_decode import BlueprintDecodeError, decode_blueprint_string


def encode_blueprint(obj: dict, version: str = "0") -> str:
    raw = json.dumps(obj).encode("utf-8")
    return version + base64.b64encode(zlib.compress(raw)).decode("ascii")


SAMPLE = {
    "blueprint": {
        "item": "blueprint",
        "entities": [
            {"entity_number": 1, "name": "transport-belt", "position": {"x": 0.5, "y": 0.5}, "direction": 2},
        ],
        "version": 281479275151360,
    }
}


def test_normal_blueprint_string():
    s = encode_blueprint(SAMPLE)
    out = decode_blueprint_string(s)
    assert out == SAMPLE
    assert out["blueprint"]["entities"][0]["name"] == "transport-belt"


def test_raw_json_fallback():
    s = json.dumps(SAMPLE)
    out = decode_blueprint_string(s)
    assert out == SAMPLE


def test_empty_string_raises():
    with pytest.raises(BlueprintDecodeError):
        decode_blueprint_string("")
    with pytest.raises(BlueprintDecodeError):
        decode_blueprint_string("   ")


def test_none_raises():
    with pytest.raises(BlueprintDecodeError):
        decode_blueprint_string(None)


def test_invalid_string_raises():
    # Looks like a versioned string but the payload is not valid base64+zlib.
    with pytest.raises(BlueprintDecodeError):
        decode_blueprint_string("0not-valid-base64-zlib-$$$")


def test_invalid_json_lookalike_raises():
    with pytest.raises(BlueprintDecodeError):
        decode_blueprint_string("{not json}")


def test_round_trip_with_version_byte_present():
    s = encode_blueprint(SAMPLE)
    assert s[0] == "0"
    assert decode_blueprint_string(s)["blueprint"]["entities"][0]["direction"] == 2
