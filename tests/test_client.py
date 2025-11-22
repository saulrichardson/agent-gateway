from gateway.client import build_user_message


def test_build_user_message_with_image(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-bytes")

    message = build_user_message(
        "Describe the screenshot.",
        image_paths=[str(image_path)],
    )

    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert message["content"][0]["type"] == "input_text"
    image_chunk = message["content"][1]
    assert image_chunk["type"] == "input_image"
    assert "image_url" in image_chunk
    assert image_chunk["image_url"].startswith("data:image")


def test_build_user_message_text_only():
    message = build_user_message("hi there")
    assert message["content"] == "hi there"
