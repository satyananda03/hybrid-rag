import asyncio

from hybrid_rag import progress


class _Sink:
    def __init__(self):
        self.events = []

    def stage(self, name):
        self.events.append(("stage", name, progress.current_doc_id.get(), progress.current_workspace.get()))

    def skip(self, name, reason):
        self.events.append(("skip", name, reason))

    def activity(self, message, level="info"):
        self.events.append(("activity", message, level))


def test_no_sink_is_a_noop():
    progress.set_sink(None)
    progress.stage("parse"), progress.skip("x", "y"), progress.activity("z")


def test_sink_receives_events_with_the_library_set_context():
    sink = _Sink()
    progress.set_sink(sink)
    try:
        async def run():
            progress.current_doc_id.set("doc-1")
            progress.current_workspace.set("ws")
            progress.stage("chunk")
            progress.skip("multimodal", "tidak ada gambar")
            progress.activity("menunggu", level="warning")

        asyncio.run(run())
    finally:
        progress.set_sink(None)
    assert sink.events == [
        ("stage", "chunk", "doc-1", "ws"),
        ("skip", "multimodal", "tidak ada gambar"),
        ("activity", "menunggu", "warning"),
    ]
