"""The 41 CUAD clause types and the mapping from CUAD questions to them."""

from __future__ import annotations

import re

CLAUSE_TYPES: list[str] = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

# Longest first, so "Affiliate License-Licensor" is not matched as a shorter name.
_BY_LENGTH = sorted(CLAUSE_TYPES, key=len, reverse=True)


def clause_type_from_question(question: str) -> str | None:
    """Return the clause type named in a CUAD question, or None."""
    for quoted in re.findall(r"[\"']([^\"']+)[\"']", question):
        for name in _BY_LENGTH:
            if quoted.strip().lower() == name.lower():
                return name
    lowered = question.lower()
    for name in _BY_LENGTH:
        if name.lower() in lowered:
            return name
    return None
