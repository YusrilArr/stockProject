import json
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

FAKE = json.dumps({"variants": [
    {"title": "Misty rice terraces", "prompt": "A farmer walking through Bali rice terraces at sunrise ...", "negative_prompt": "text, logo, blurry"},
    {"title": "Close-up hands", "prompt": "Close-up of weathered hands planting rice seedlings ...", "negative_prompt": "watermark"},
]})

def test_pages_render():
    for p in ["views/home.py", "views/upscaler.py", "views/metadata.py", "views/prompt_generator.py"]:
        at = AppTest.from_file("../" + p).run()
        assert not at.exception, (p, at.exception)

def test_prompt_flow():
    with patch("prompt_gen.generator.generate_text", return_value=FAKE) as m:
        at = AppTest.from_file("../views/prompt_generator.py").run()
        at.sidebar.text_input[0].input("dummy-key")
        at.text_area[0].input("sawah terasering Bali pagi hari")
        at.button[0].click().run()
        assert not at.exception
        assert not at.error, [e.value for e in at.error]
        assert len(at.code) == 4  # 2 prompt + 2 negative
        req = m.call_args.args[0]
        assert "sawah terasering" in req and "Adobe Stock" in req

def test_empty_idea_shows_error():
    at = AppTest.from_file("../views/prompt_generator.py").run()
    at.sidebar.text_input[0].input("k")
    at.button[0].click().run()
    assert at.error and "Isi dulu" in at.error[0].value

def test_parse_fenced():
    from prompt_gen.generator import parse_variants
    r = parse_variants("```json\n" + FAKE + "\n```")
    assert len(r) == 2 and r[0].negative_prompt == "text, logo, blurry"

def test_app_navigation():
    at = AppTest.from_file("../app.py").run()
    assert not at.exception
