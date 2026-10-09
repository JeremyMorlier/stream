"""Base of the visualization stages: pass-through stages that draw what the stages after them yield."""

import logging
from abc import abstractmethod
from typing import Any

from stream.stages.stage import Stage

logger = logging.getLogger(__name__)


class ExtraInfoReportStage(Stage):
    """A stage that changes nothing about the pipeline: it forwards its kwargs untouched to the next stage and
    yields every answer it gets back unchanged, calling `report` on each answer whose `extra_info` carries
    `report_key` along the way.

    Place it *before* the stage whose results it draws (answers travel back up the chain), e.g. right before
    `CoreArchitectureExplorationStage` in a schedule search. A failure while drawing is logged and never
    fails the pipeline: a missing figure must not cost a search's result. Subclasses read their own settings
    from the kwargs with `self.kwargs.get(...)` instead of named parameters, so nothing they configure is
    consumed before it reaches the stages that actually use it."""

    report_key = "schedule_report"

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for scme, extra_info in sub_stage.run():
            if isinstance(extra_info, dict) and self.report_key in extra_info:
                try:
                    self.report(scme, extra_info)
                except Exception:
                    logger.warning(f"{type(self).__name__}: could not write its report.", exc_info=True)
            yield scme, extra_info

    @abstractmethod
    def report(self, scme: Any, extra_info: dict[str, Any]) -> None:
        """Write this stage's artifacts for one answer."""
