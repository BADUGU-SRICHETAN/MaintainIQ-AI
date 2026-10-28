from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3


def test_investigation_flow_produces_an_answer():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.text_input[0].set_value("Why did the bearing on the drive end fail?").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("bearing" in md.value.lower() for md in at.markdown)
