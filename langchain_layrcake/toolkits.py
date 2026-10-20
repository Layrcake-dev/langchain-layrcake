"""A toolkit that gives an agent every Layrcake tool, sharing one run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from langchain_core.tools import BaseTool, BaseToolkit
from layrcake import Layrcake
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from langchain_layrcake._core import DEFAULT_BUDGET_CREDITS, LayrcakeSession
from langchain_layrcake.tools import (
    LayrcakeCheckCompanies,
    LayrcakeCreateCampaign,
    LayrcakeEnrich,
    LayrcakeFindLeads,
    LayrcakeGetMe,
    LayrcakeLaunchCampaign,
    LayrcakeResearchCompany,
    LayrcakeTool,
    LayrcakeVerifyEmail,
)

__all__ = ["LayrcakeToolkit"]

_TOOLS: tuple[Callable[..., LayrcakeTool], ...] = (
    LayrcakeFindLeads,
    LayrcakeEnrich,
    LayrcakeVerifyEmail,
    LayrcakeCheckCompanies,
    LayrcakeResearchCompany,
    LayrcakeCreateCampaign,
    LayrcakeLaunchCampaign,
    LayrcakeGetMe,
)


class LayrcakeToolkit(BaseToolkit):
    """Every Layrcake tool, sharing one run budget.

    Setup:
        Install `langchain-layrcake` and set `LAYRCAKE_API_KEY`.

    Key init args:
        budget_credits: The most this run may spend, in credits. Default 50.
        dry_run: Return each call's plan instead of running it. Charges nothing.
        api_key: Optional. Defaults to the LAYRCAKE_API_KEY environment variable.

    Instantiate:
        .. code-block:: python

            from langchain_layrcake import LayrcakeToolkit

            toolkit = LayrcakeToolkit(budget_credits=50)
            tools = toolkit.get_tools()

    Use one toolkit per agent run: once launch_campaign has asked for approval,
    its tools refuse further paid calls.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    budget_credits: float = float(DEFAULT_BUDGET_CREDITS)
    dry_run: bool = False
    api_key: Optional[SecretStr] = None
    run_id: Optional[str] = None
    require_verified: bool = True
    client: Optional[Layrcake] = Field(default=None, exclude=True)

    _session: Optional[LayrcakeSession] = PrivateAttr(default=None)

    @property
    def session(self) -> LayrcakeSession:
        """The run state the toolkit's tools share."""
        if self._session is None:
            self._session = LayrcakeSession(
                self.client,
                api_key=self.api_key.get_secret_value() if self.api_key else None,
                budget_credits=self.budget_credits,
                dry_run=self.dry_run,
                run_id=self.run_id,
                require_verified=self.require_verified,
            )
        return self._session

    def get_tools(self) -> list[BaseTool]:
        """Return the eight Layrcake tools, all bound to this toolkit's run."""
        return [tool(session=self.session) for tool in _TOOLS]
