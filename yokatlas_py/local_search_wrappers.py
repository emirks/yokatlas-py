"""
Local search wrappers for YOKATLAS data stored in JSON files.
This module provides fast searching without API calls by using locally cached data.
"""

import json
import os
import glob
import random
import math
from typing import Any, Dict, List, Optional, Union
from .search_utils import (
    normalize_search_params,
    expand_program_name,
    find_best_university_match,
    normalize_university_name,
    normalize_score_type,
)


class LocalYOKATLASSearcher:
    """Local searcher for YOKATLAS data using JSON files."""

    def __init__(self, data_directory: str = None):
        """
        Initialize the local searcher.

        Args:
            data_directory: Path to directory containing JSON data files
        """
        if data_directory is None:
            # Default to the yokatlas-results directory relative to this file
            current_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(current_dir)
            self.data_directory = os.path.join(parent_dir, "yokatlas-results")
        else:
            self.data_directory = data_directory

        self._data_cache = {}  # Cache loaded data

    def _get_data_files(self, program_type: str = "lisans") -> Dict[str, str]:
        """
        Get available data files for the specified program type.

        Args:
            program_type: "lisans" or "onlisans"

        Returns:
            Dictionary mapping score types to file paths
        """
        files = {}

        # Look for the most recent data directory based on program type
        if program_type == "onlisans":
            pattern = os.path.join(self.data_directory, "all_onlisans_programs_*")
        else:
            pattern = os.path.join(self.data_directory, "all_programs_*")

        data_dirs = glob.glob(pattern)

        if not data_dirs:
            # Fallback: try generic pattern for both types
            fallback_pattern = os.path.join(self.data_directory, "all_*programs_*")
            all_dirs = glob.glob(fallback_pattern)

            # Filter directories that contain the right program type files
            for dir_path in all_dirs:
                test_pattern = os.path.join(dir_path, f"{program_type}_programs_*.json")
                if glob.glob(test_pattern):
                    data_dirs.append(dir_path)

        if not data_dirs:
            return files

        # Use the most recent directory (sorted by name)
        latest_dir = sorted(data_dirs)[-1]

        # Find files for the specified program type
        file_pattern = os.path.join(latest_dir, f"{program_type}_programs_*.json")
        data_files = glob.glob(file_pattern)

        for file_path in data_files:
            filename = os.path.basename(file_path)
            # Extract score type from filename (e.g., "lisans_programs_ea.json" -> "ea")
            if "_" in filename:
                parts = filename.replace(".json", "").split("_")
                if len(parts) >= 3:
                    score_type = parts[-1]
                    files[score_type] = file_path

        return files

    def _load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file with caching.

        Args:
            file_path: Path to JSON file

        Returns:
            List of program data
        """
        if file_path in self._data_cache:
            return self._data_cache[file_path]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._data_cache[file_path] = data
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading data from {file_path}: {e}")
            return []

    def _match_text(self, text: str, query: str) -> bool:
        """
        Check if text matches query (case-insensitive, partial match).

        Args:
            text: Text to search in
            query: Query to search for

        Returns:
            True if query is found in text
        """
        if not text or not query:
            return True if not query else False

        return query.lower() in text.lower()

    def _match_single_filter(
        self,
        program: Dict[str, Any],
        filter_key: str,
        filter_value: Any,
        program_field: str,
    ) -> bool:
        """
        Check if a program matches a single filter criterion.

        Args:
            program: Program data dictionary
            filter_key: The filter key name
            filter_value: Single value or list of values to match
            program_field: Field name in the program data to match against

        Returns:
            True if program matches the filter
        """
        if isinstance(filter_value, list):
            # Match if ANY value in the list matches
            return any(
                self._match_text(program.get(program_field, ""), value)
                for value in filter_value
            )
        else:
            # Single value match
            return self._match_text(program.get(program_field, ""), filter_value)

    def _match_program_filter(
        self, program: Dict[str, Any], program_filter: Any
    ) -> bool:
        """
        Check if a program matches the program name filter (handles combined program fields).

        Args:
            program: Program data dictionary
            program_filter: Single program name or list of program names

        Returns:
            True if program matches the filter
        """
        program_name = program.get("program_adi", "")
        program_detail = program.get("program_detay", "")
        combined_program = f"{program_name} {program_detail}"

        if isinstance(program_filter, list):
            # Match if ANY program name in the list matches
            return any(
                self._match_text(combined_program, prog) for prog in program_filter
            )
        else:
            # Single program filter
            return self._match_text(combined_program, program_filter)

    def _match_ranking_filters(
        self, program: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """
        Check if a program matches ranking range filters.

        Args:
            program: Program data dictionary
            filters: Search filters containing ust_bs and/or alt_bs

        Returns:
            True if program matches ranking filters
        """
        if "ust_bs" not in filters and "alt_bs" not in filters:
            return True

        # Get 2024 TBS (most recent available)
        tbs_2024 = program.get("tbs", {}).get("2024")
        if not tbs_2024 or tbs_2024 == "---":
            return True

        try:
            tbs_value = int(tbs_2024)

            # Upper limit (better ranking - lower number)
            if "ust_bs" in filters and filters["ust_bs"]:
                ust_bs = int(filters["ust_bs"])
                if tbs_value < ust_bs:  # Better than upper limit
                    return False

            # Lower limit (worse ranking - higher number)
            if "alt_bs" in filters and filters["alt_bs"]:
                alt_bs = int(filters["alt_bs"])
                if tbs_value > alt_bs:  # Worse than lower limit
                    return False

            return True
        except (ValueError, TypeError):
            return True

    def _match_program(self, program: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a program matches the given filters.

        Args:
            program: Program data dictionary
            filters: Search filters (supports list inputs for universite, program, sehir)

        Returns:
            True if program matches all filters
        """
        # University filter - supports list input
        if "universite" in filters and filters["universite"]:
            if not self._match_single_filter(
                program, "universite", filters["universite"], "uni_adi"
            ):
                return False

        # Program filter - supports list input (special handling for combined fields)
        if "program" in filters and filters["program"]:
            if not self._match_program_filter(program, filters["program"]):
                return False

        # City filter - supports list input
        if "sehir" in filters and filters["sehir"]:
            if not self._match_single_filter(
                program, "sehir", filters["sehir"], "sehir_adi"
            ):
                return False

        # Simple single-value filters
        simple_filters = {
            "universite_turu": "universite_turu",
            "ucret": "ucret_burs",
            "ogretim_turu": "ogretim_turu",
        }

        for filter_key, program_field in simple_filters.items():
            if filter_key in filters and filters[filter_key]:
                if not self._match_text(
                    program.get(program_field, ""), filters[filter_key]
                ):
                    return False

        # Ranking range filters
        if not self._match_ranking_filters(program, filters):
            return False

        return True

    def _prepare_search_parameters(
        self, params: Dict[str, Any], program_type: str, smart_search: bool
    ) -> tuple[Dict[str, Any], Optional[int]]:
        """
        Prepare and normalize search parameters.

        Args:
            params: Raw search parameters
            program_type: "lisans" or "onlisans"
            smart_search: Enable smart parameter normalization

        Returns:
            Tuple of (normalized_params, original_siralama)
        """
        if smart_search:
            normalized_params = normalize_search_params(params, program_type)
        else:
            normalized_params = params.copy()

        # Handle siralama parameter (convert to ust_bs and alt_bs)
        original_siralama = None
        if "siralama" in normalized_params or "sıralama" in normalized_params:
            siralama = normalized_params.get("siralama") or normalized_params.get(
                "sıralama"
            )
            if siralama:
                try:
                    siralama_int = int(siralama)
                    original_siralama = siralama_int
                    # Calculate range based on siralama * 0.5 to siralama * 1.5
                    better_ranking = int(siralama_int * 0.5)
                    worse_ranking = int(siralama_int * 1.5)
                    normalized_params["ust_bs"] = str(better_ranking)
                    normalized_params["alt_bs"] = str(worse_ranking)

                    # Remove siralama params as they've been converted
                    normalized_params.pop("siralama", None)
                    normalized_params.pop("sıralama", None)
                except (ValueError, TypeError):
                    pass

        return normalized_params, original_siralama

    def _get_score_type(
        self, normalized_params: Dict[str, Any], program_type: str
    ) -> str:
        """
        Determine the score type for the search.

        Args:
            normalized_params: Normalized search parameters
            program_type: "lisans" or "onlisans"

        Returns:
            Score type string
        """
        if program_type == "onlisans":
            return "tyt"  # Onlisans programs only use TYT score type
        else:
            return normalize_score_type(normalized_params.get("puan_turu", "say"))

    def _expand_program_list(
        self, program_query: Union[str, List[str]], program_type: str
    ) -> List[str]:
        """
        Expand program name(s) to all possible variations.

        Args:
            program_query: Single program name or list of program names
            program_type: "lisans" or "onlisans"

        Returns:
            List of all expanded program variations
        """
        if isinstance(program_query, list):
            # For lists, expand each program name and collect all variations
            all_program_variations = []
            for single_program in program_query:
                program_variations = expand_program_name(single_program, program_type)
                all_program_variations.extend(program_variations)

            # Remove duplicates while preserving order
            seen = set()
            unique_variations = []
            for variation in all_program_variations:
                if variation not in seen:
                    seen.add(variation)
                    unique_variations.append(variation)
            return unique_variations
        else:
            # Single program string
            return expand_program_name(program_query, program_type)

    def _search_with_program_expansion(
        self,
        programs: List[Dict[str, Any]],
        normalized_params: Dict[str, Any],
        program_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Search programs with program name expansion enabled.

        Args:
            programs: List of all programs to search through
            normalized_params: Normalized search parameters
            program_type: "lisans" or "onlisans"

        Returns:
            List of matching programs
        """
        results = []
        program_query = normalized_params["program"]
        program_variations = self._expand_program_list(program_query, program_type)

        # Try each program variation
        for program_name in program_variations:
            search_params_variant = normalized_params.copy()
            search_params_variant["program"] = program_name

            # Search with this variant
            for program in programs:
                if self._match_program(program, search_params_variant):
                    # Avoid duplicates
                    if not any(
                        r.get("yop_kodu") == program.get("yop_kodu") for r in results
                    ):
                        results.append(program)

        return results

    def _search_without_expansion(
        self, programs: List[Dict[str, Any]], normalized_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search programs without program name expansion.

        Args:
            programs: List of all programs to search through
            normalized_params: Normalized search parameters

        Returns:
            List of matching programs
        """
        results = []
        for program in programs:
            if self._match_program(program, normalized_params):
                results.append(program)
        return results

    def _calculate_median_tbs(self, results: List[Dict[str, Any]]) -> Optional[int]:
        """
        Calculate median TBS value from results.

        Args:
            results: List of program results

        Returns:
            Median TBS value or None if no valid TBS values found
        """
        valid_tbs_values = []
        for program in results:
            tbs_2024 = program.get("tbs", {}).get("2024")
            if tbs_2024 and tbs_2024 != "---":
                try:
                    valid_tbs_values.append(int(tbs_2024))
                except (ValueError, TypeError):
                    continue

        if valid_tbs_values:
            valid_tbs_values.sort()
            return valid_tbs_values[len(valid_tbs_values) // 2]
        return None

    def _apply_result_sampling(
        self,
        results: List[Dict[str, Any]],
        max_results: int,
        original_siralama: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Apply bell curve sampling or truncation to results if needed.

        Args:
            results: List of search results
            max_results: Maximum number of results to return
            original_siralama: Original siralama parameter if provided

        Returns:
            Sampled or truncated results
        """
        if len(results) <= max_results:
            return results

        if original_siralama is not None:
            # Use specified siralama as center
            return self._sample_bell_curve(results, original_siralama, max_results)
        else:
            # No siralama specified - use median TBS as center for bell curve
            median_tbs = self._calculate_median_tbs(results)
            if median_tbs:
                return self._sample_bell_curve(results, median_tbs, max_results)
            else:
                # Fallback to simple truncation if no valid TBS values
                return results[:max_results]

    def _sample_bell_curve(
        self, programs: List[Dict[str, Any]], target_ranking: int, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Sample programs using a bell curve distribution centered at target_ranking.

        Args:
            programs: List of programs to sample from
            target_ranking: Center of the bell curve (siralama value)
            max_results: Maximum number of results to return

        Returns:
            Sampled programs distributed in a bell curve around target_ranking
        """
        if len(programs) <= max_results:
            return programs

        # Filter programs that have valid TBS values and calculate distances
        valid_programs = []
        for program in programs:
            tbs_2024 = program.get("tbs", {}).get("2024")
            if tbs_2024 and tbs_2024 != "---":
                try:
                    tbs_value = int(tbs_2024)
                    distance = abs(tbs_value - target_ranking)
                    valid_programs.append((program, tbs_value, distance))
                except (ValueError, TypeError):
                    continue

        if not valid_programs:
            return programs[:max_results]

        # Sort by distance from target ranking
        valid_programs.sort(key=lambda x: x[2])  # Sort by distance

        # Calculate bell curve weights using normal distribution
        # Standard deviation is 1/3 of the range to make a nice bell curve
        std_dev = target_ranking * 0.3  # Adjust this to control curve width
        weights = []

        for program, tbs_value, distance in valid_programs:
            # Calculate normal distribution weight
            weight = math.exp(-0.5 * ((tbs_value - target_ranking) / std_dev) ** 2)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Fallback to uniform weights
            weights = [1.0 / len(valid_programs)] * len(valid_programs)

        # Sample programs based on weights
        try:
            # Use weighted random sampling
            sampled_indices = random.choices(
                range(len(valid_programs)),
                weights=weights,
                k=min(max_results, len(valid_programs)),
            )

            # Remove duplicates while preserving order
            seen_indices = set()
            unique_indices = []
            for idx in sampled_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    unique_indices.append(idx)

            # If we don't have enough unique samples, fill with closest programs
            while len(unique_indices) < max_results and len(unique_indices) < len(
                valid_programs
            ):
                for i, (program, tbs_value, distance) in enumerate(valid_programs):
                    if i not in seen_indices:
                        unique_indices.append(i)
                        seen_indices.add(i)
                        if len(unique_indices) >= max_results:
                            break

            # Get the sampled programs
            sampled_programs = [
                valid_programs[i][0] for i in unique_indices[:max_results]
            ]

            # Sort by TBS value for better presentation
            sampled_programs.sort(
                key=lambda p: int(p.get("tbs", {}).get("2024", "999999"))
            )

            return sampled_programs

        except Exception as e:
            # Fallback to simple closest selection
            return [prog[0] for prog in valid_programs[:max_results]]

    def search_programs(
        self,
        params: Dict[str, Any],
        program_type: str = "lisans",
        smart_search: bool = True,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for programs in local data.

        Args:
            params: Search parameters
            program_type: "lisans" or "onlisans"
            smart_search: Enable smart parameter normalization
            max_results: Maximum number of results to return (applies bell curve sampling if exceeded)

        Returns:
            List of matching programs
        """
        normalized_params, original_siralama = self._prepare_search_parameters(
            params, program_type, smart_search
        )

        # Determine score type based on program type
        score_type = self._get_score_type(normalized_params, program_type)

        # Get data files
        data_files = self._get_data_files(program_type)

        if score_type not in data_files:
            print(f"No data file found for score type: {score_type}")
            return []

        # Load data
        programs = self._load_data(data_files[score_type])

        if not programs:
            return []

        results = []

        # Handle program expansion if smart search is enabled
        if (
            smart_search
            and "program" in normalized_params
            and normalized_params["program"]
        ):
            results = self._search_with_program_expansion(
                programs, normalized_params, program_type
            )
        else:
            # Regular search without program expansion
            results = self._search_without_expansion(programs, normalized_params)

        # Apply bell curve sampling if we have too many results
        results = self._apply_result_sampling(results, max_results, original_siralama)

        return results

    def format_search_results(
        self,
        results: List[Dict[str, Any]],
        search_params: Dict[str, Any],
        original_siralama: Optional[int] = None,
        max_results: int = 100,
    ) -> str:
        """
        Format search results into a human-readable string.

        Args:
            results: List of program results
            search_params: Original search parameters
            original_siralama: Original siralama parameter if provided
            max_results: Maximum results requested

        Returns:
            Formatted string representation of results
        """
        if not results:
            return "No programs found matching your criteria."

        # Determine sampling type
        sampling_info = ""
        if len(results) == max_results and len(results) > 10:
            if original_siralama:
                sampling_info = f" (bell curve centered at ranking {original_siralama})"
            else:
                # Calculate median for display
                valid_tbs = []
                for program in results:
                    tbs_2024 = program.get("tbs", {}).get("2024")
                    if tbs_2024 and tbs_2024 != "---":
                        try:
                            valid_tbs.append(int(tbs_2024))
                        except:
                            pass
                if valid_tbs:
                    median_tbs = sorted(valid_tbs)[len(valid_tbs) // 2]
                    sampling_info = f" (bell curve centered at median TBS {median_tbs})"

        # Build header
        header = f"Found {len(results)} programs{sampling_info}:\n"

        # Format each result
        formatted_lines = []
        for i, program in enumerate(results, 1):
            uni_name = program.get("uni_adi", "N/A")
            program_name = program.get("program_adi", "N/A")
            program_detail = program.get("program_detay", "")
            tbs_2024 = program.get("tbs", {}).get("2024", "N/A")
            taban_2024 = program.get("taban", {}).get("2024", "N/A")
            city = program.get("sehir_adi", "N/A")
            uni_type = program.get("universite_turu", "N/A")
            yop_kodu = program.get("yop_kodu", "N/A")

            # Main line
            main_line = f"     {i}. {uni_name} - {program_name}"
            if program_detail and program_detail.strip():
                main_line += f" {program_detail}"

            # Details line
            details = []
            if tbs_2024 != "N/A":
                details.append(f"Taban Sıralama: {tbs_2024}")
            if taban_2024 != "N/A":
                details.append(f"Taban Puanı: {taban_2024}")
            details.append(f"Şehir: {city}")
            details.append(f"Üniversite Türü: {uni_type}")
            details.append(f"YOP Kodu: {yop_kodu}")

            details_line = f"        {' | '.join(details)}"

            formatted_lines.append(main_line)
            formatted_lines.append(details_line)

        return header + "\n".join(formatted_lines)


def search_local_lisans_programs(
    params: Dict[str, Any],
    smart_search: bool = True,
    data_directory: str = None,
    max_results: int = 100,
    return_formatted: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enhanced local search for lisans (bachelor's) programs.

    Args:
        params: Search parameters (can use common variations, supports lists for universite/program/sehir)
        smart_search: Enable smart parameter normalization and program expansion
        data_directory: Path to directory containing JSON data files
        max_results: Maximum number of results (applies bell curve sampling if exceeded with siralama)
        return_formatted: If True, returns dict with both results and formatted string

    Returns:
        List of programs found, or dict with results and formatted string if return_formatted=True

    Example:
        >>> # These all work:
        >>> search_local_lisans_programs({"uni_adi": "Boğaziçi", "program_adi": "Bilgisayar"})
        >>> search_local_lisans_programs({"universite": "ODTÜ", "program": "Yazılım"})
        >>> search_local_lisans_programs({"uni": "itü", "bolum": "elektrik", "puan_turu": "SAY"})
        >>> search_local_lisans_programs({"siralama": 1000})  # Filter by ranking range
        >>> search_local_lisans_programs({"siralama": 1000, "sehir": "istanbul"}, max_results=50, return_formatted=True)  # With formatting
        >>> # List inputs for multiple values:
        >>> search_local_lisans_programs({"universite": ["Boğaziçi", "ODTÜ"], "program": ["bilgisayar", "yazılım"]})
        >>> search_local_lisans_programs({"sehir": ["istanbul", "ankara"], "program": ["mühendislik", "tıp"]})
    """
    searcher = LocalYOKATLASSearcher(data_directory)

    # Store original siralama for formatting
    original_siralama = params.get("siralama") or params.get("sıralama")
    if original_siralama:
        try:
            original_siralama = int(original_siralama)
        except (ValueError, TypeError):
            original_siralama = None

    results = searcher.search_programs(params, "lisans", smart_search, max_results)

    if return_formatted:
        formatted_string = searcher.format_search_results(
            results, params, original_siralama, max_results
        )
        return {
            "results": results,
            "formatted": formatted_string,
            "total_found": len(results),
        }

    return results


def search_local_onlisans_programs(
    params: Dict[str, Any],
    smart_search: bool = True,
    data_directory: str = None,
    max_results: int = 100,
    return_formatted: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enhanced local search for onlisans (associate) programs.

    Args:
        params: Search parameters (can use common variations including sıralama for bell curve sampling, supports lists for universite/program/sehir)
        smart_search: Enable smart parameter normalization and program expansion
        data_directory: Path to directory containing JSON data files
        max_results: Maximum number of results (applies bell curve sampling if exceeded with sıralama)
        return_formatted: If True, returns dict with both results and formatted string

    Returns:
        List of programs found, or dict with results and formatted string if return_formatted=True

    Example:
        >>> # These all work for onlisans:
        >>> search_local_onlisans_programs({"universite": "ANADOLU", "program": "Bilgisayar"})
        >>> search_local_onlisans_programs({"sıralama": 5000, "sehir": "istanbul"}, max_results=50, return_formatted=True)
        >>> search_local_onlisans_programs({"program": "Turizm", "universite_turu": "Devlet"})
        >>> # List inputs for multiple values:
        >>> search_local_onlisans_programs({"universite": ["ANADOLU", "ISTANBUL"], "program": ["turizm", "otel"]})
        >>> search_local_onlisans_programs({"sehir": ["istanbul", "izmir"], "program": ["bilgisayar", "elektronik"]})
    """
    searcher = LocalYOKATLASSearcher(data_directory)

    # Store original siralama for formatting (onlisans DOES use siralama!)
    original_siralama = params.get("siralama") or params.get("sıralama")
    if original_siralama:
        try:
            original_siralama = int(original_siralama)
        except (ValueError, TypeError):
            original_siralama = None

    results = searcher.search_programs(params, "onlisans", smart_search, max_results)

    if return_formatted:
        formatted_string = searcher.format_search_results(
            results, params, original_siralama, max_results
        )
        return {
            "results": results,
            "formatted": formatted_string,
            "total_found": len(results),
        }

    return results


def search_local_programs(
    params: Dict[str, Any],
    program_type: Optional[str] = None,
    smart_search: bool = True,
    data_directory: str = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for programs in local data, supporting both lisans and onlisans.

    Args:
        params: Search parameters
        program_type: "lisans", "onlisans", or None for both
        smart_search: Enable smart parameter normalization
        data_directory: Path to directory containing JSON data files

    Returns:
        Dictionary with program types as keys and lists of programs as values
    """
    searcher = LocalYOKATLASSearcher(data_directory)
    results = {}

    if program_type is None:
        # Search both types
        results["lisans"] = searcher.search_programs(params, "lisans", smart_search)
        results["onlisans"] = searcher.search_programs(params, "onlisans", smart_search)
    elif program_type in ["lisans", "onlisans"]:
        results[program_type] = searcher.search_programs(
            params, program_type, smart_search
        )
    else:
        raise ValueError("program_type must be 'lisans', 'onlisans', or None")

    return results
