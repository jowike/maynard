from maynard.pipelines.default import pipeline


def test_pipeline_registry():
    assert pipeline is not None
