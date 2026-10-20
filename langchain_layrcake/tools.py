"""LangChain tools for the Layrcake API, one per verb.

Each tool validates its arguments with a Pydantic model and hands them to a
shared `LayrcakeSession`, which calls the official `layrcake` SDK and enforces
the run rules: the run budget, dry_run, idempotency keys, verified-only
campaigns and a launch that stops at a person's approval.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from langchain_layrcake._core import SPECS, LayrcakeSession, ToolSpec, invalid_arguments

__all__ = [
    "LayrcakeCheckCompanies",
    "LayrcakeCreateCampaign",
    "LayrcakeEnrich",
    "LayrcakeFindLeads",
    "LayrcakeGetMe",
    "LayrcakeLaunchCampaign",
    "LayrcakeResearchCompany",
    "LayrcakeTool",
    "LayrcakeVerifyEmail",
]


def _readable_validation_error(error: Union[ValidationError, ValidationErrorV1]) -> str:
    if isinstance(error, ValidationErrorV1):
        return f"Refused: invalid arguments: {error}"
    return invalid_arguments(error)


class LayrcakeTool(BaseTool):
    """Base class for the Layrcake tools. Use the subclasses or `LayrcakeToolkit`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ClassVar[ToolSpec]
    session: LayrcakeSession = Field(default_factory=LayrcakeSession, exclude=True)
    """The run state shared by the tools of one run. Built from LAYRCAKE_API_KEY when omitted."""

    handle_validation_error: Union[bool, str, Callable[[Union[ValidationError, ValidationErrorV1]], str], None] = (
        _readable_validation_error
    )

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        arguments = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in kwargs.items()}
        return self.session.call(self.spec.name, arguments)


def _spec(name: str) -> ToolSpec:
    return SPECS[name]


class LayrcakeFindLeads(LayrcakeTool):
    """Find people who match an audience. 0.25 credits per person found, 3 minimum."""

    spec: ClassVar[ToolSpec] = _spec("find_leads")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeEnrich(LayrcakeTool):
    """Find one person's work email. 5 credits when found."""

    spec: ClassVar[ToolSpec] = _spec("enrich")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeVerifyEmail(LayrcakeTool):
    """Check that an address can receive mail. 1.25 credits."""

    spec: ClassVar[ToolSpec] = _spec("verify_email")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeCheckCompanies(LayrcakeTool):
    """Answer one yes or no question per company. 3.5 credits per company answered."""

    spec: ClassVar[ToolSpec] = _spec("check_companies")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeResearchCompany(LayrcakeTool):
    """Research one company from public sources. 9 credits per completed research."""

    spec: ClassVar[ToolSpec] = _spec("research_company")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeCreateCampaign(LayrcakeTool):
    """Save a draft campaign for verified people. Drafts never send. Free."""

    spec: ClassVar[ToolSpec] = _spec("create_campaign")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeLaunchCampaign(LayrcakeTool):
    """Ask a person to approve a draft. Sends nothing and returns an approval link. Free."""

    spec: ClassVar[ToolSpec] = _spec("launch_campaign")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args


class LayrcakeGetMe(LayrcakeTool):
    """Show the workspace, its credits and the run budget. Free."""

    spec: ClassVar[ToolSpec] = _spec("get_me")
    name: str = spec.name
    description: str = spec.description
    args_schema: type[BaseModel] = spec.args
