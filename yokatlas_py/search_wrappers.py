"""Enhanced search wrappers with fuzzy matching and smart parameter handling."""

from typing import Any, Optional, Union
from .lisanstercihsihirbazi import YOKATLASLisansTercihSihirbazi
from .onlisanstercihsihirbazi import YOKATLASOnlisansTercihSihirbazi
from .search_utils import normalize_search_params, expand_program_name


def search_lisans_programs(
    params: dict[str, Any], smart_search: bool = True
) -> list[dict[str, Any]]:
    """
    Enhanced search for lisans (bachelor's) programs.

    Args:
        params: Search parameters (can use common variations)
        smart_search: Enable smart parameter normalization and program expansion

    Returns:
        List of programs found, or empty list if none

    Example:
        >>> # These all work now:
        >>> search_lisans_programs({"uni_adi": "Boğaziçi", "program_adi": "Bilgisayar"})
        >>> search_lisans_programs({"universite": "ODTÜ", "program": "Yazılım"})
        >>> search_lisans_programs({"uni": "itü", "bolum": "elektrik", "puan_turu": "SAY"})
        >>> search_lisans_programs({"siralama": 1000})  # Filter by ranking range
    """
    if not smart_search:
        # Use original search without enhancements
        search = YOKATLASLisansTercihSihirbazi(params)
        results = search.search()
    else:
        # Normalize parameters
        normalized_params = normalize_search_params(params, "lisans")

        # If program is specified, try expanding it
        if "program" in normalized_params:
            program_query = normalized_params["program"]
            program_variations = expand_program_name(program_query, "lisans")

            all_results = []

            # Try each program variation
            for program_name in program_variations:
                search_params_variant = normalized_params.copy()
                search_params_variant["program"] = program_name

                search = YOKATLASLisansTercihSihirbazi(search_params_variant)
                results = search.search()

                if isinstance(results, list) and len(results) > 0:
                    # Add results, avoiding duplicates
                    for result in results:
                        # Check if this program is already in results (by program code)
                        if not any(
                            r.get("yop_kodu") == result.get("yop_kodu")
                            for r in all_results
                        ):
                            all_results.append(result)

            results = all_results
        else:
            # No program specified, just do normal search
            search = YOKATLASLisansTercihSihirbazi(normalized_params)
            results = search.search()
            results = results if isinstance(results, list) else []

    return results if isinstance(results, list) else []


def search_onlisans_programs(
    params: dict[str, Any], smart_search: bool = True
) -> list[dict[str, Any]]:
    """
    Enhanced search for önlisans (associate) programs.

    Args:
        params: Search parameters (can use common variations)
        smart_search: Enable smart parameter normalization and program expansion

    Returns:
        List of programs found, or empty list if none
    """
    if not smart_search:
        # Use original search without enhancements
        search = YOKATLASOnlisansTercihSihirbazi(params)
        return search.search()

    # Normalize parameters (note: önlisans uses "tyt" instead of score types)
    normalized_params = normalize_search_params(params, "onlisans")

    # Ensure correct score type for önlisans
    if "puan_turu" in normalized_params and normalized_params["puan_turu"] in [
        "say",
        "ea",
        "söz",
        "dil",
    ]:
        normalized_params["puan_turu"] = "tyt"

    # If program is specified, try expanding it
    if "program" in normalized_params:
        program_query = normalized_params["program"]
        program_variations = expand_program_name(program_query, "onlisans")

        all_results = []

        # Try each program variation
        for program_name in program_variations:
            search_params = normalized_params.copy()
            search_params["program"] = program_name

            search = YOKATLASOnlisansTercihSihirbazi(search_params)
            results = search.search()

            if isinstance(results, list) and len(results) > 0:
                # Add results, avoiding duplicates
                for result in results:
                    # Check if this program is already in results (by program code)
                    if not any(
                        r.get("yop_kodu") == result.get("yop_kodu") for r in all_results
                    ):
                        all_results.append(result)

        return all_results
    else:
        # No program specified, just do normal search
        search = YOKATLASOnlisansTercihSihirbazi(normalized_params)
        results = search.search()
        return results if isinstance(results, list) else []


def search_programs(
    params: dict[str, Any],
    program_type: Optional[str] = None,
    smart_search: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Universal search for both lisans and önlisans programs.

    Args:
        params: Search parameters
        program_type: "lisans", "onlisans", or None for both
        smart_search: Enable smart parameter normalization

    Returns:
        Dictionary with "lisans" and/or "onlisans" keys containing results

    Example:
        >>> results = search_programs({"uni": "boğaziçi", "program": "bilgisayar"})
        >>> print(f"Found {len(results['lisans'])} lisans programs")
        >>> print(f"Found {len(results['onlisans'])} önlisans programs")
    """
    results = {}

    if program_type is None or program_type == "lisans":
        results["lisans"] = search_lisans_programs(params, smart_search)

    if program_type is None or program_type == "onlisans":
        results["onlisans"] = search_onlisans_programs(params, smart_search)

    return results


def search_all_lisans_programs(
    params: dict[str, Any], smart_search: bool = True
) -> list[dict[str, Any]]:
    """
    Search ALL lisans programs by iterating through pagination until all results are retrieved.

    Args:
        params: Search parameters (can use common variations)
        smart_search: Enable smart parameter normalization and program expansion

    Returns:
        Complete list of all programs found across all pages

    Example:
        >>> # Get ALL programs matching criteria
        >>> all_programs = search_all_lisans_programs({"universite": "Boğaziçi"})
        >>> print(f"Found {len(all_programs)} total programs")
    """
    all_results = []
    start = 0
    length = 100  # Use 100 as requested

    print(f"Starting search with pagination (length={length})...")

    while True:
        # Create params for this page
        page_params = params.copy()
        page_params["start"] = start
        page_params["length"] = length

        print(f"Fetching results {start}-{start+length-1}...")

        if smart_search:
            # Use the enhanced search with normalization
            normalized_params = normalize_search_params(page_params, "lisans")

            # If program is specified, try expanding it (only on first page to avoid duplicates)
            if "program" in normalized_params and start == 0:
                program_query = normalized_params["program"]
                program_variations = expand_program_name(program_query, "lisans")

                page_results = []
                # Try each program variation
                for program_name in program_variations:
                    search_params_variant = normalized_params.copy()
                    search_params_variant["program"] = program_name

                    # For program variations, we need to paginate each one
                    variation_start = 0
                    while True:
                        search_params_variant["start"] = variation_start
                        search_params_variant["length"] = length

                        search = YOKATLASLisansTercihSihirbazi(search_params_variant)
                        variation_results = search.search()

                        if (
                            not isinstance(variation_results, list)
                            or len(variation_results) == 0
                        ):
                            break

                        # Add results, avoiding duplicates
                        for result in variation_results:
                            if not any(
                                r.get("yop_kodu") == result.get("yop_kodu")
                                for r in page_results
                            ):
                                page_results.append(result)

                        # If we got fewer results than requested, we've reached the end
                        if len(variation_results) < length:
                            break

                        variation_start += length

                # Add all unique results from this page
                for result in page_results:
                    if not any(
                        r.get("yop_kodu") == result.get("yop_kodu") for r in all_results
                    ):
                        all_results.append(result)

                # For program expansion, we've already gotten all results
                break
            else:
                # Normal search without program expansion
                search = YOKATLASLisansTercihSihirbazi(normalized_params)
                page_results = search.search()
        else:
            # Use original search without enhancements
            search = YOKATLASLisansTercihSihirbazi(page_params)
            page_results = search.search()

        # Check if we got valid results
        if not isinstance(page_results, list):
            print(f"Error or invalid response: {page_results}")
            break

        if len(page_results) == 0:
            print("No more results found.")
            break

        # Add results to our collection
        all_results.extend(page_results)
        print(f"Added {len(page_results)} results. Total so far: {len(all_results)}")

        # If we got fewer results than requested, we've reached the end
        if len(page_results) < length:
            print("Reached end of results (partial page).")
            break

        # Move to next page
        start += length

    print(f"Search completed. Total results: {len(all_results)}")
    return all_results


def search_all_onlisans_programs(
    params: dict[str, Any], smart_search: bool = True
) -> list[dict[str, Any]]:
    """
    Search ALL önlisans programs by iterating through pagination until all results are retrieved.

    Args:
        params: Search parameters (can use common variations)
        smart_search: Enable smart parameter normalization and program expansion

    Returns:
        Complete list of all programs found across all pages
    """
    all_results = []
    start = 0
    length = 100  # Use 100 as requested

    print(f"Starting önlisans search with pagination (length={length})...")

    while True:
        # Create params for this page
        page_params = params.copy()
        page_params["start"] = start
        page_params["length"] = length

        print(f"Fetching önlisans results {start}-{start+length-1}...")

        if smart_search:
            # Normalize parameters (note: önlisans uses "tyt" instead of score types)
            normalized_params = normalize_search_params(page_params, "onlisans")

            # Ensure correct score type for önlisans
            if "puan_turu" in normalized_params and normalized_params["puan_turu"] in [
                "say",
                "ea",
                "söz",
                "dil",
            ]:
                normalized_params["puan_turu"] = "tyt"

            # If program is specified, try expanding it (only on first page to avoid duplicates)
            if "program" in normalized_params and start == 0:
                program_query = normalized_params["program"]
                program_variations = expand_program_name(program_query, "onlisans")

                page_results = []
                # Try each program variation
                for program_name in program_variations:
                    search_params_variant = normalized_params.copy()
                    search_params_variant["program"] = program_name

                    # For program variations, we need to paginate each one
                    variation_start = 0
                    while True:
                        search_params_variant["start"] = variation_start
                        search_params_variant["length"] = length

                        search = YOKATLASOnlisansTercihSihirbazi(search_params_variant)
                        variation_results = search.search()

                        if (
                            not isinstance(variation_results, list)
                            or len(variation_results) == 0
                        ):
                            break

                        # Add results, avoiding duplicates
                        for result in variation_results:
                            if not any(
                                r.get("yop_kodu") == result.get("yop_kodu")
                                for r in page_results
                            ):
                                page_results.append(result)

                        # If we got fewer results than requested, we've reached the end
                        if len(variation_results) < length:
                            break

                        variation_start += length

                # Add all unique results from this page
                for result in page_results:
                    if not any(
                        r.get("yop_kodu") == result.get("yop_kodu") for r in all_results
                    ):
                        all_results.append(result)

                # For program expansion, we've already gotten all results
                break
            else:
                # Normal search without program expansion
                search = YOKATLASOnlisansTercihSihirbazi(normalized_params)
                page_results = search.search()
        else:
            # Use original search without enhancements
            search = YOKATLASOnlisansTercihSihirbazi(page_params)
            page_results = search.search()

        # Check if we got valid results
        if not isinstance(page_results, list):
            print(f"Error or invalid response: {page_results}")
            break

        if len(page_results) == 0:
            print("No more results found.")
            break

        # Add results to our collection
        all_results.extend(page_results)
        print(f"Added {len(page_results)} results. Total so far: {len(all_results)}")

        # If we got fewer results than requested, we've reached the end
        if len(page_results) < length:
            print("Reached end of results (partial page).")
            break

        # Move to next page
        start += length

    print(f"Önlisans search completed. Total results: {len(all_results)}")
    return all_results


def search_all_programs(
    params: dict[str, Any],
    program_type: Optional[str] = None,
    smart_search: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Universal search for ALL programs (both lisans and önlisans) with complete pagination.

    Args:
        params: Search parameters
        program_type: "lisans", "onlisans", or None for both
        smart_search: Enable smart parameter normalization

    Returns:
        Dictionary with "lisans" and/or "onlisans" keys containing ALL results

    Example:
        >>> # Get ALL programs matching criteria
        >>> all_results = search_all_programs({"universite": "Boğaziçi"})
        >>> print(f"Found {len(all_results['lisans'])} total lisans programs")
        >>> print(f"Found {len(all_results['onlisans'])} total önlisans programs")
    """
    results = {}

    if program_type is None or program_type == "lisans":
        results["lisans"] = search_all_lisans_programs(params, smart_search)

    if program_type is None or program_type == "onlisans":
        results["onlisans"] = search_all_onlisans_programs(params, smart_search)

    return results
