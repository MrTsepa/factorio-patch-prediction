import numpy as np

from factorio_patches.blueprint_extract import extract_blueprints
from factorio_patches.rasterize import rasterize_blueprint
from factorio_patches.vocab import EMPTY_ID, UNK_ID, Vocab, entity_token


def make_vocab(tokens):
    return Vocab(["EMPTY", "MASK", "UNK"] + tokens)


def test_normalization_and_placement():
    vocab = make_vocab([entity_token("transport-belt", 2), entity_token("fast-inserter", 4)])
    bp = {
        "entities": [
            {"name": "transport-belt", "position": {"x": 10, "y": 5}, "direction": 2},
            {"name": "fast-inserter", "position": {"x": 12, "y": 5}, "direction": 4},
        ]
    }
    rb = rasterize_blueprint(bp, vocab)
    # min x=10 -> col 0; min y=5 -> row 0; width spans 10..12 => 3
    assert rb.height == 1
    assert rb.width == 3
    assert rb.grid[0, 0] == vocab.encode(entity_token("transport-belt", 2))
    assert rb.grid[0, 2] == vocab.encode(entity_token("fast-inserter", 4))
    assert rb.grid[0, 1] == EMPTY_ID
    assert rb.n_entities == 2
    assert rb.n_cells_filled == 2
    assert rb.n_collisions == 0


def test_rounding_half_up_and_negative():
    vocab = make_vocab([entity_token("pipe", 0)])
    bp = {"entities": [
        {"name": "pipe", "position": {"x": -0.5, "y": -0.5}},
        {"name": "pipe", "position": {"x": 0.5, "y": 0.5}},
    ]}
    rb = rasterize_blueprint(bp, vocab)
    # -0.5 -> 0, 0.5 -> 1 after round-half-up; normalized min -> 0
    assert rb.height == 2 and rb.width == 2
    assert rb.grid[0, 0] == vocab.encode(entity_token("pipe", 0))
    assert rb.grid[1, 1] == vocab.encode(entity_token("pipe", 0))


def test_collision_counting():
    vocab = make_vocab([entity_token("transport-belt", 0)])
    bp = {"entities": [
        {"name": "transport-belt", "position": {"x": 0, "y": 0}},
        {"name": "transport-belt", "position": {"x": 0.2, "y": -0.2}},  # rounds to same cell
    ]}
    rb = rasterize_blueprint(bp, vocab)
    assert rb.n_entities == 2
    assert rb.n_cells_filled == 1
    assert rb.n_collisions == 1


def test_unknown_entity_becomes_unk():
    vocab = make_vocab([entity_token("transport-belt", 0)])
    bp = {"entities": [{"name": "nuclear-reactor", "position": {"x": 0, "y": 0}}]}
    rb = rasterize_blueprint(bp, vocab)
    assert rb.grid[0, 0] == UNK_ID
    assert rb.n_unk == 1


def test_extract_then_rasterize():
    decoded = {"blueprint_book": {"label": "b", "blueprints": [
        {"blueprint": {"entities": [{"name": "transport-belt", "position": {"x": 0, "y": 0}, "direction": 2}]}},
        {"upgrade_planner": {"settings": {}}},  # must be skipped
        {"blueprint": {"entities": []}},          # empty, dropped
    ]}}
    bps = extract_blueprints(decoded, source_hash="abc")
    assert len(bps) == 1
    vocab = make_vocab([entity_token("transport-belt", 2)])
    rb = rasterize_blueprint(bps[0], vocab)
    assert rb.n_cells_filled == 1
