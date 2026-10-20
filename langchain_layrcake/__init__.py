"""LangChain tools for Layrcake, go-to-market infrastructure for AI agents."""

from importlib import metadata

from layrcake.agent_tools import LayrcakeSession

from langchain_layrcake.toolkits import LayrcakeToolkit
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

try:
    __version__ = metadata.version(__package__ or "langchain_layrcake")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "LayrcakeCheckCompanies",
    "LayrcakeCreateCampaign",
    "LayrcakeEnrich",
    "LayrcakeFindLeads",
    "LayrcakeGetMe",
    "LayrcakeLaunchCampaign",
    "LayrcakeResearchCompany",
    "LayrcakeSession",
    "LayrcakeTool",
    "LayrcakeToolkit",
    "LayrcakeVerifyEmail",
    "__version__",
]
