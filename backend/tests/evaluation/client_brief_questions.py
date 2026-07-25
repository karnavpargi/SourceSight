from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteClass = Literal["extractive", "synthesis", "boundary"]


@dataclass(frozen=True)
class ClientBriefCase:
    question: str
    expected_route: RouteClass


# These strings must match docs/client-brief.md verbatim (and are mirrored in
# frontend/src/lib/chat-prompts.ts).
APPLE_REVENUE_MIX = (
    "Across Apple's 2021–2025 10-Ks, how did the revenue mix between iPhone, Services, "
    "Mac, iPad, and Wearables change, and which category appears to have contributed "
    "most to any mix shift?"
)

AMAZON_SEGMENTS = (
    "For Amazon, compare AWS operating income and margin against North America and "
    "International from 2021–2025. In which years did AWS appear to fund losses or "
    "weaker profitability elsewhere?"
)

NVIDIA_DEMAND_DRIVERS = (
    "How did NVIDIA describe demand drivers, customer concentration, and supply "
    "constraints for its Data Center business from fiscal 2021 through fiscal 2025?"
)

MICROSOFT_AZURE_LANGUAGE = (
    "Across Microsoft's 2021–2025 filings, what changed in the way the company "
    "describes Azure, AI infrastructure, and cloud capacity constraints?"
)

ALPHABET_REVENUE_TRENDS = (
    "For Alphabet, how did Google Search, YouTube ads, Google Network, "
    "subscriptions/platforms/devices, and Google Cloud revenue trends differ across "
    "the available 10-Ks?"
)

FIVE_COMPANY_RISK_FACTORS = (
    "Which of the five companies added, removed, or materially changed risk-factor "
    "language related to AI, cloud infrastructure, export controls, supply chain "
    "concentration, or regulation between 2021 and 2025?"
)

APPLE_NVIDIA_SUPPLIERS = (
    "For Apple and NVIDIA, what do the filings say about supplier concentration or "
    "dependence on third-party manufacturing, and did the wording become more or less "
    "urgent over time?"
)

FOUR_COMPANY_CAPEX = (
    "Compare capital expenditures and purchase commitments for Microsoft, Alphabet, "
    "Amazon, and NVIDIA. What do the filings imply about the scale and timing of "
    "AI/cloud infrastructure investment?"
)

GEOGRAPHIC_EXPOSURE = (
    "For each company, summarize the most important geographic revenue exposures "
    "disclosed in the latest 10-K, then identify any year-over-year changes that "
    "could matter to an analyst."
)

GENAI_MARGIN_PROOF = (
    "If an analyst asks whether the filings prove that generative AI improved margins "
    "for any of these companies, what evidence exists in the corpus, and where should "
    "the bot refuse to infer beyond the filings?"
)


CLIENT_BRIEF_CASES: tuple[ClientBriefCase, ...] = (
    ClientBriefCase(APPLE_REVENUE_MIX, "extractive"),
    ClientBriefCase(AMAZON_SEGMENTS, "extractive"),
    ClientBriefCase(NVIDIA_DEMAND_DRIVERS, "synthesis"),
    ClientBriefCase(MICROSOFT_AZURE_LANGUAGE, "synthesis"),
    ClientBriefCase(ALPHABET_REVENUE_TRENDS, "extractive"),
    ClientBriefCase(FIVE_COMPANY_RISK_FACTORS, "synthesis"),
    ClientBriefCase(APPLE_NVIDIA_SUPPLIERS, "synthesis"),
    ClientBriefCase(FOUR_COMPANY_CAPEX, "extractive"),
    ClientBriefCase(GEOGRAPHIC_EXPOSURE, "extractive"),
    ClientBriefCase(GENAI_MARGIN_PROOF, "boundary"),
)
