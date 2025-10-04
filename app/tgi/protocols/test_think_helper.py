from app.tgi.protocols.chunk_reader import create_response_chunk
from app.tgi.protocols.think_helper import ThinkExtractor


def test_example_sequence():
    ex = ThinkExtractor(id_generator=lambda: "test_id")
    inputs = [
        "bla",
        "<th",
        "ink",
        ">",
        "hello world</t",
        "hink>",
        "bla",
        "<think>he",
        "llo again</think>",
        "bla",
        "<think>",
    ]
    expected = [
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        create_response_chunk("test_id", "<think>hello world</think>"),
        "\n",
        "\n",
        create_response_chunk("test_id", "<think>hello again</think>"),
        "\n",
        "\n",
    ]

    outputs = [ex.feed(s) for s in inputs]
    assert outputs == expected


def test_multiple_blocks_in_one_chunk():
    ex = ThinkExtractor(id_generator=lambda: "test_id")
    chunk = "prefix<think>A</think>mid<think>B</think>suffix"
    out1 = ex.feed(chunk)
    assert out1 == create_response_chunk("test_id", "<think>A</think><think>B</think>")


def test_partial_opening_tag_preserved():
    ex = ThinkExtractor(id_generator=lambda: "test_id")
    assert ex.feed("<th") == ""
    assert ex.feed("ink>content</think>") == create_response_chunk(
        "test_id", "<think>content</think>"
    )
