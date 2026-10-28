"""Starts the API with a stand-in detector (hand-annotated boxes) so the full HTTP surface can be
exercised on machines without a trained model. NOT for production."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
import main  # noqa: E402
from pipeline.engine import Engine  # noqa: E402
from tests.test_rules import INDIA_1  # noqa: E402


class Fixed:
    model_names = ["annotated-demo"]
    def __call__(self, bgr):
        return list(INDIA_1)


main.build_engine = lambda: Engine(Fixed(), main.SETTINGS, store=main.store, notify=main.notify_threadsafe)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(main.app, host="127.0.0.1", port=8000)
