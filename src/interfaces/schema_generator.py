"""
Interface — Schema Generator.

Dynamically generates a JSON schema based on query semantics.
"""

from abc import ABC, abstractmethod


class ISchemaGenerator(ABC):
    """Contract for dynamic schema generation."""

    @abstractmethod
    def generate_schema(self, user_query: str) -> list[dict]:
        """
        Analyse the user query and return a list of field definitions.

        Mandatory fields (source_link, title, main_content) are always
        included. Additional fields are inferred from query semantics.

        Returns
        -------
        list[dict]
            Each dict has: {"name": str, "type": str, "description": str}
        """
        ...
