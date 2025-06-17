"""Project pipelines."""

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from maynard.pipelines.default import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipeline = create_pipeline()
    return {
        # "data_processing": pipeline,
        "__default__": pipeline,
    }
