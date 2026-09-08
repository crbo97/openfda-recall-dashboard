
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import re
import shutil
import tempfile
import textwrap
import time
import unicodedata
import zipfile
import numpy
import pandas
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(page_title="FDA Medical Device Recall Topic Explorer", layout="wide")


string_app_dir = Path(__file__).resolve().parent
string_dashboard_source_code_path = Path(__file__).resolve()
string_default_project_root = string_app_dir.parent

EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST_FILE = "experiment_manifest.json"
MAX_EXPERIMENT_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_EXPERIMENT_UNCOMPRESSED_BYTES = 1_500 * 1024 * 1024
MAX_EXPERIMENT_ARCHIVE_MEMBERS = 500

object_app_logger = logging.getLogger(__name__)


def build_experiment_paths(string_project_root):
    string_project_root = Path(string_project_root).resolve()
    string_results_dir = string_project_root / "reports" / "bertopic_unsupervised_mpnet"
    string_interim_dir = string_project_root / "data" / "interim"

    return {
        "project_root": string_project_root,
        "dashboard_records": string_project_root / "dashboard" / "data" / "dashboard_topic_records.parquet",
        "manual_labels": string_results_dir / "manual_topic_labels.csv",
        "topic_keywords": string_results_dir / "bertopic_unsupervised_topic_keywords.parquet",
        "representative_documents": string_results_dir / "bertopic_unsupervised_representative_documents.parquet",
        "document_topics": string_interim_dir / "bertopic_unsupervised_document_topics.parquet",
        "embeddings": string_interim_dir / "reason_for_recall_all_mpnet_base_v2_embeddings.npy",
        "record_mapping": string_interim_dir / "reason_for_recall_record_embedding_map.parquet",
        "semantic_search_index": string_results_dir / "semantic_search_index.parquet",
        "semantic_search_config": string_results_dir / "semantic_search_config.json",
        "manufacturer_terms": string_results_dir / "semantic_search_manufacturer_terms.json",
        "experiment_metadata": string_results_dir / "bertopic_unsupervised_experiment_metadata.json",
        "cluster_summary": string_results_dir / "topic_cluster_summary.csv",
        "static_figures": string_project_root / "outputs" / "figures" / "dissertation_static",
        "manifest": string_project_root / EXPERIMENT_MANIFEST_FILE,
    }


string_custom_experiment_root_value = st.session_state.get("custom_experiment_root")

boolean_custom_experiment_is_available = bool(string_custom_experiment_root_value and Path(string_custom_experiment_root_value).exists())

if not boolean_custom_experiment_is_available:
    if "custom_experiment_root" in st.session_state:
        del st.session_state["custom_experiment_root"]
    st.session_state["experiment_source_selector"] = "Default experiment"

list_experiment_source_options = ["Default experiment"]

if boolean_custom_experiment_is_available:
    list_experiment_source_options.append("Custom experiment")

if st.session_state.get("activate_custom_experiment", False):
    st.session_state["experiment_source_selector"] = "Custom experiment"
    st.session_state["activate_custom_experiment"] = False

string_selected_experiment_source = st.sidebar.radio(
    "Experiment data",
    options=list_experiment_source_options,
    key="experiment_source_selector",
    help=(
        "Switch between the dissertation experiment and the most "
        "recently generated or uploaded custom experiment."
    ),
)

if string_selected_experiment_source == "Custom experiment":
    string_active_project_root = Path(st.session_state["custom_experiment_root"]).resolve()
else:
    string_active_project_root = string_default_project_root

dictionary_active_experiment_paths = build_experiment_paths(string_active_project_root)

string_project_root = string_active_project_root
string_dashboard_records_path = dictionary_active_experiment_paths["dashboard_records"]
string_manual_labels_path = dictionary_active_experiment_paths["manual_labels"]
string_topic_keywords_path = dictionary_active_experiment_paths["topic_keywords"]
string_representative_documents_path = dictionary_active_experiment_paths["representative_documents"]
string_embeddings_path = dictionary_active_experiment_paths["embeddings"]
string_record_mapping_path = dictionary_active_experiment_paths["record_mapping"]
string_semantic_search_index_path = dictionary_active_experiment_paths["semantic_search_index"]
string_semantic_search_config_path = dictionary_active_experiment_paths["semantic_search_config"]
string_manufacturer_terms_path = dictionary_active_experiment_paths["manufacturer_terms"]
string_experiment_metadata_path = dictionary_active_experiment_paths["experiment_metadata"]
string_static_figures_dir = dictionary_active_experiment_paths["static_figures"]

list_required_paths = [
    string_dashboard_records_path,
    string_manual_labels_path,
    string_topic_keywords_path,
    string_representative_documents_path,
    string_embeddings_path,
    string_record_mapping_path,
    string_semantic_search_index_path,
    string_semantic_search_config_path,
    string_manufacturer_terms_path,
]

if string_selected_experiment_source == "Custom experiment":
    list_required_paths.append(dictionary_active_experiment_paths["manifest"])

list_missing_paths = []

for string_required_path in list_required_paths:
    if not string_required_path.exists():
        list_missing_paths.append(string_required_path)

if list_missing_paths:
    st.error(
        "The dashboard cannot find one or more saved analysis "
        "artifacts. Re-run the notebook's Streamlit section."
    )
    list_missing_path_strings = []

    for string_missing_path in list_missing_paths:
        list_missing_path_strings.append(str(string_missing_path))

    st.code("\n".join(list_missing_path_strings))
    st.stop()


@st.cache_data(show_spinner=False)
def load_dashboard_data(
    string_dashboard_records_path,
    string_manual_labels_path,
    string_topic_keywords_path,
    string_representative_documents_path,
):
    data_frame_full_records = pandas.read_parquet(string_dashboard_records_path)

    data_frame_manual_labels = pandas.read_csv(string_manual_labels_path)

    data_frame_topic_keywords = pandas.read_parquet(string_topic_keywords_path)

    data_frame_representative_documents = pandas.read_parquet(string_representative_documents_path)

    return (data_frame_full_records, data_frame_manual_labels, data_frame_topic_keywords, data_frame_representative_documents)


(data_frame_records, data_frame_manual_labels, data_frame_topic_keywords, data_frame_representative_documents) = load_dashboard_data(
    str(string_dashboard_records_path),
    str(string_manual_labels_path),
    str(string_topic_keywords_path),
    str(string_representative_documents_path),
)


data_frame_manual_labels["topic"] = data_frame_manual_labels["topic"].astype(int)

dictionary_topic_label_lookup = data_frame_manual_labels.set_index("topic")["manual_label"].to_dict()

data_frame_records["bertopic_Topic"] = data_frame_records["bertopic_Topic"].astype(int)

data_frame_records["topic_label"] = (data_frame_records["bertopic_Topic"].map(dictionary_topic_label_lookup).fillna("Unlabelled topic"))

data_frame_records["event_date_initiated"] = (pandas.to_datetime(data_frame_records["event_date_initiated"], errors="coerce"))

data_frame_records["year_initiated"] = data_frame_records["event_date_initiated"].dt.year.astype("Int64")

data_frame_records["recalling_firm_clean"] = (
    data_frame_records["recalling_firm_clean"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .fillna("Missing / not available")
)

data_frame_records["device_class_clean"] = (data_frame_records["device_class_clean"].astype("string").fillna("Missing / not available"))

if "occurrence_count" not in data_frame_records.columns:
    series_narrative_occurrence_lookup = data_frame_records["semantic_text_id"].value_counts()

    data_frame_records["occurrence_count"] = (data_frame_records["semantic_text_id"].map(series_narrative_occurrence_lookup).astype(int))

integer_number_missing_labels = int((data_frame_records["topic_label"] == "Unlabelled topic").sum())


def format_topic(integer_topic_id):
    string_topic_label = dictionary_topic_label_lookup.get(int(integer_topic_id), "Unlabelled topic")

    return f"{int(integer_topic_id)} - {string_topic_label}"


def calculate_yearly_topic_trend(data_frame_context, integer_topic_id, boolean_use_unique_narratives):
    if boolean_use_unique_narratives:
        series_yearly_totals = (data_frame_context.groupby("year_initiated")["semantic_text_id"].nunique().rename("all_record_count"))

        series_yearly_topic_counts = (
            data_frame_context[data_frame_context["bertopic_Topic"] == integer_topic_id]
            .groupby("year_initiated")["semantic_text_id"]
            .nunique()
            .rename("topic_record_count")
        )
    else:
        series_yearly_totals = data_frame_context.groupby("year_initiated").size().rename("all_record_count")

        series_yearly_topic_counts = (
            data_frame_context[data_frame_context["bertopic_Topic"] == integer_topic_id]
            .groupby("year_initiated")
            .size()
            .rename("topic_record_count")
        )

    data_frame_trend = (
        pandas.concat([series_yearly_totals, series_yearly_topic_counts], axis=1)
        .fillna({"topic_record_count": 0})
        .reset_index()
    )

    data_frame_trend["topic_record_count"] = data_frame_trend["topic_record_count"].astype(int)

    data_frame_trend["topic_share_percent"] = (100 * data_frame_trend["topic_record_count"] / data_frame_trend["all_record_count"])

    return data_frame_trend.sort_values("year_initiated")


def calculate_topic_prevalence_by_group(data_frame_context, integer_topic_id, string_group_column, boolean_use_unique_narratives):
    if boolean_use_unique_narratives:
        series_group_totals = (
            data_frame_context
            .groupby(string_group_column, dropna=False)["semantic_text_id"]
            .nunique()
            .rename("all_record_count")
        )

        series_topic_counts = (
            data_frame_context[data_frame_context["bertopic_Topic"] == integer_topic_id]
            .groupby(string_group_column, dropna=False)["semantic_text_id"]
            .nunique()
            .rename("topic_record_count")
        )
    else:
        series_group_totals = (data_frame_context.groupby(string_group_column, dropna=False).size().rename("all_record_count"))

        series_topic_counts = (
            data_frame_context[data_frame_context["bertopic_Topic"] == integer_topic_id]
            .groupby(string_group_column, dropna=False)
            .size()
            .rename("topic_record_count")
        )

    data_frame_result = (
        pandas.concat([series_group_totals, series_topic_counts], axis=1)
        .fillna({"topic_record_count": 0})
        .reset_index()
    )

    data_frame_result["topic_record_count"] = data_frame_result["topic_record_count"].astype(int)

    data_frame_result["topic_prevalence_percent"] = (100 * data_frame_result["topic_record_count"] / data_frame_result["all_record_count"])

    return data_frame_result


@st.cache_data(show_spinner=False)
def load_semantic_search_data(
    string_semantic_search_index_path,
    string_embeddings_path,
    string_record_mapping_path,
    string_semantic_search_config_path,
    string_manufacturer_terms_path,
):
    data_frame_similarity_index = pandas.read_parquet(string_semantic_search_index_path)

    array_corpus_embeddings = numpy.load(string_embeddings_path, allow_pickle=False)

    data_frame_record_mapping = pandas.read_parquet(string_record_mapping_path)

    with Path(string_semantic_search_config_path).open("r", encoding="utf-8") as object_file:
        dictionary_search_config = json.load(object_file)

    with Path(string_manufacturer_terms_path).open("r", encoding="utf-8") as object_file:
        list_manufacturer_terms = json.load(object_file)

    return (
        data_frame_similarity_index,
        array_corpus_embeddings,
        data_frame_record_mapping,
        dictionary_search_config,
        list_manufacturer_terms,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model(string_model_name):
    from sentence_transformers import SentenceTransformer

    try:
        model_embedding = SentenceTransformer(string_model_name, local_files_only=True)
    except OSError:
        model_embedding = SentenceTransformer(string_model_name)

    return model_embedding


def validate_semantic_search_artifacts(
    data_frame_similarity_index,
    array_corpus_embeddings,
    data_frame_record_mapping,
    dictionary_search_config,
):
    set_required_index_columns = {
        "semantic_text_id",
        "semantic_text",
        "occurrence_count",
        "embedding_row",
        "bertopic_topic",
        "manual_topic_label",
    }

    set_missing_index_columns = set_required_index_columns.difference(data_frame_similarity_index.columns)

    if set_missing_index_columns:
        string_missing_index_columns = ", ".join(sorted(set_missing_index_columns))
        raise ValueError("The semantic-search index is missing required columns: " + string_missing_index_columns)

    if array_corpus_embeddings.ndim != 2:
        raise ValueError("The semantic-search embeddings must be a two-dimensional array.")

    if len(data_frame_similarity_index) != len(array_corpus_embeddings):
        raise ValueError("The semantic-search index and embedding array must have the same number of rows.")

    if not data_frame_similarity_index["embedding_row"].is_unique:
        raise ValueError("Each semantic-search embedding row must be unique.")

    if not numpy.array_equal(data_frame_similarity_index["embedding_row"].to_numpy(), numpy.arange(len(data_frame_similarity_index))):
        raise ValueError("The semantic-search index rows are not aligned with the embedding array.")

    if (array_corpus_embeddings.shape[1] != dictionary_search_config["embedding_dimension"]):
        raise ValueError("The saved embedding dimension does not match the search configuration.")

    if dictionary_search_config["normalize_embeddings"] is not True:
        raise ValueError("The semantic-search embeddings must use normalization.")

    if not data_frame_record_mapping["cfres_id"].is_unique:
        raise ValueError("Each FDA recall record identifier must be unique in the record mapping.")

    set_unknown_semantic_text_ids = set(data_frame_record_mapping["semantic_text_id"]).difference(set(data_frame_similarity_index["semantic_text_id"]))

    if set_unknown_semantic_text_ids:
        raise ValueError("The record mapping contains semantic text identifiers that are absent from the search index.")


def normalise_recall_text_for_embedding(variable_value):
    if pandas.isna(variable_value):
        return ""

    string_text = str(variable_value)
    string_text = unicodedata.normalize("NFKC", string_text)
    string_text = re.sub(r"\s+", " ", string_text)

    return string_text.strip()


def mask_query_manufacturer_name(string_text, string_recalling_firm, list_manufacturer_terms):
    if string_recalling_firm is None or not str(string_recalling_firm).strip():
        return string_text

    string_firm_text = normalise_recall_text_for_embedding(string_recalling_firm).lower()

    list_terms_to_mask = []

    for string_manufacturer_term in list_manufacturer_terms:
        if (isinstance(string_manufacturer_term, str) and string_manufacturer_term.lower() in string_firm_text):
            list_terms_to_mask.append(string_manufacturer_term.lower())

    list_terms_to_mask.append(string_firm_text)
    string_cleaned_text = string_text

    for string_manufacturer_term in sorted(set(list_terms_to_mask), key=len, reverse=True):
        if string_manufacturer_term:
            string_cleaned_text = string_cleaned_text.replace(string_manufacturer_term, "manufacturer")

    string_cleaned_text = string_cleaned_text.replace("manufacturer inc.", "manufacturer")

    return string_cleaned_text


def remove_generic_query_words(string_text, list_masked_words):
    string_cleaned_text = string_text

    for string_masked_word in list_masked_words:
        string_pattern = rf"\b{re.escape(string_masked_word)}\b"
        string_cleaned_text = re.sub(string_pattern, "", string_cleaned_text, flags=re.IGNORECASE)

    return normalise_recall_text_for_embedding(string_cleaned_text)


def prepare_reason_for_recall_query(string_reason_for_recall, string_recalling_firm, list_manufacturer_terms, list_masked_words):
    string_query_text = normalise_recall_text_for_embedding(string_reason_for_recall).lower()

    string_query_text = mask_query_manufacturer_name(
        string_text=string_query_text,
        string_recalling_firm=string_recalling_firm,
        list_manufacturer_terms=list_manufacturer_terms,
    )

    string_query_text = remove_generic_query_words(string_text=string_query_text, list_masked_words=list_masked_words)

    return string_query_text


def embed_semantic_search_query(string_prepared_query, model_embedding):
    array_query_embedding = model_embedding.encode(
        [string_prepared_query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return array_query_embedding.astype(numpy.float32, copy=False)[0]


def find_nearest_semantic_narratives(
    string_prepared_query,
    model_embedding,
    data_frame_similarity_index,
    array_corpus_embeddings,
    integer_number_of_results=10,
    boolean_include_complex_narratives=True,
):
    array_query_embedding = embed_semantic_search_query(string_prepared_query=string_prepared_query, model_embedding=model_embedding)

    data_frame_search = data_frame_similarity_index.copy()

    if not boolean_include_complex_narratives:
        data_frame_search = data_frame_search[data_frame_search["bertopic_topic"] != -1].copy()

    array_embedding_rows = data_frame_search["embedding_row"].to_numpy(dtype=numpy.int32)

    data_frame_search["cosine_similarity"] = array_corpus_embeddings[array_embedding_rows] @ array_query_embedding

    data_frame_nearest_narratives = (
        data_frame_search
        .sort_values("cosine_similarity", ascending=False)
        .head(integer_number_of_results)
        .reset_index(drop=True)
    )

    data_frame_nearest_narratives.insert(0, "similarity_rank", numpy.arange(1, len(data_frame_nearest_narratives) + 1))

    data_frame_nearest_narratives["prepared_query_text"] = string_prepared_query

    return data_frame_nearest_narratives


def expand_semantic_matches_to_records(data_frame_nearest_narratives, data_frame_record_mapping, data_frame_full_recall_records):
    list_identifier_columns = ["cfres_id", "product_res_number", "res_event_number"]

    data_frame_record_links = (data_frame_record_mapping[list_identifier_columns + ["semantic_text_id"]].drop_duplicates())

    list_metadata_columns = [
        "cfres_id",
        "product_res_number",
        "res_event_number",
        "event_date_initiated",
        "recall_status",
        "recalling_firm",
        "firm_fei_number",
        "state",
        "product_code",
        "device_name",
        "medical_specialty_description",
        "device_class",
        "device_class_clean",
        "product_description",
        "root_cause_description",
        "reason_for_recall",
    ]

    list_available_metadata_columns = []

    for string_column in list_metadata_columns:
        if string_column in data_frame_full_recall_records.columns:
            list_available_metadata_columns.append(string_column)

    data_frame_record_metadata = data_frame_full_recall_records[list_available_metadata_columns].copy()

    data_frame_full_matches = (
        data_frame_nearest_narratives
        .merge(data_frame_record_links, on="semantic_text_id", how="left", validate="one_to_many")
        .merge(data_frame_record_metadata, on=list_identifier_columns, how="left", validate="many_to_one")
        .sort_values(["similarity_rank", "event_date_initiated"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return data_frame_full_matches


OPENFDA_RECALL_ENDPOINT = "https://api.fda.gov/device/recall.json"
OPENFDA_PAGE_LIMIT = 1_000
OPENFDA_REQUEST_TIMEOUT_SECONDS = 120
OPENFDA_REQUEST_PAUSE_SECONDS = 0.30
OPENFDA_YEAR_MAX_ATTEMPTS = 10
OPENFDA_YEAR_RETRY_WAIT_SECONDS = 60
EXPERIMENT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EXPERIMENT_RANDOM_SEED = 50
MINIMUM_UNIQUE_NARRATIVES = 100

EXPERIMENT_STATIC_FIGURE_STEMS = [
    "figure_16_01_topic_prevalence",
    "figure_16_02_temporal_prevalence",
    "figure_16_03_temporal_change",
    "figure_16_04_topics_by_device_class",
    "figure_16_05_firm_topic_profiles",
    "figure_16_06_complex_narratives_temporal",
    "figure_16_07_record_event_sensitivity",
    "figure_16_08_semantic_topic_map",
]

EXPERIMENT_REQUIRED_RELATIVE_PATHS = [
    "dashboard/data/dashboard_topic_records.parquet",
    (
        "data/interim/"
        "reason_for_recall_all_mpnet_base_v2_embeddings.npy"
    ),
    (
        "data/interim/"
        "reason_for_recall_record_embedding_map.parquet"
    ),
    (
        "data/interim/"
        "bertopic_unsupervised_document_topics.parquet"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "manual_topic_labels.csv"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "bertopic_unsupervised_topic_keywords.parquet"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "bertopic_unsupervised_representative_documents.parquet"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "semantic_search_index.parquet"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "semantic_search_config.json"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "semantic_search_manufacturer_terms.json"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "bertopic_unsupervised_experiment_metadata.json"
    ),
    (
        "reports/bertopic_unsupervised_mpnet/"
        "topic_cluster_summary.csv"
    ),
]

list_static_figure_file_extensions = ["png", "pdf"]

for string_figure_stem in EXPERIMENT_STATIC_FIGURE_STEMS:
    for string_file_extension in list_static_figure_file_extensions:
        EXPERIMENT_REQUIRED_RELATIVE_PATHS.append("outputs/figures/dissertation_static/" + f"{string_figure_stem}.{string_file_extension}")

PIPELINE_GENERIC_MASKED_WORDS = [
    "recall",
    "recalling",
    "recalled",
    "voluntarily",
    "voluntary",
    "initiated",
    "initiating",
    "issue",
    "problem",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

for integer_year in range(2010, 2031):
    PIPELINE_GENERIC_MASKED_WORDS.append(str(integer_year))


def update_experiment_progress(function_progress_callback, float_fraction, string_stage):
    if function_progress_callback is not None:
        function_progress_callback(min(max(float(float_fraction), 0.0), 1.0), str(string_stage))


def calculate_file_sha256(string_path):
    object_file_digest = hashlib.sha256()

    with Path(string_path).open("rb") as object_file:
        while True:
            bytes_file_block = object_file.read(1024 * 1024)

            if not bytes_file_block:
                break

            object_file_digest.update(bytes_file_block)

    return object_file_digest.hexdigest()


def resolve_openfda_api_key(string_api_key, function_progress_callback=None, function_request_get=None):
    string_cleaned_api_key = str(string_api_key).strip() if string_api_key else ""

    if not string_cleaned_api_key:
        return None

    if function_request_get is None:
        function_request_get = requests.get

    try:
        object_openfda_response = function_request_get(
            OPENFDA_RECALL_ENDPOINT,
            params={"limit": 1, "api_key": string_cleaned_api_key},
            timeout=OPENFDA_REQUEST_TIMEOUT_SECONDS,
        )
        object_openfda_response.raise_for_status()
        dictionary_response_payload = object_openfda_response.json()

        if not isinstance(dictionary_response_payload.get("results"), list):
            raise RuntimeError("The FDA key check returned no results.")

        return string_cleaned_api_key

    except Exception:
        object_app_logger.warning(
            "The supplied FDA API key was not accepted. "
            "Continuing without a key."
        )
        update_experiment_progress(function_progress_callback, 0.01, "FDA API key not accepted. Continuing without a key.")

        return None


def fetch_openfda_recall_year(integer_year, string_api_key=None, function_request_get=None, function_sleep=None):
    if function_request_get is None:
        function_request_get = requests.get

    if function_sleep is None:
        function_sleep = time.sleep

    string_search_query = (
        f"event_date_initiated:[{int(integer_year)}0101 TO "
        f"{int(integer_year)}1231]"
    )
    list_all_records = []
    integer_expected_total = None
    integer_skip = 0

    while integer_expected_total is None or integer_skip < integer_expected_total:
        dictionary_request_parameters = {"search": string_search_query, "limit": OPENFDA_PAGE_LIMIT, "skip": integer_skip}

        if string_api_key:
            dictionary_request_parameters["api_key"] = string_api_key

        object_openfda_response = function_request_get(
            OPENFDA_RECALL_ENDPOINT,
            params=dictionary_request_parameters,
            timeout=OPENFDA_REQUEST_TIMEOUT_SECONDS,
        )
        object_openfda_response.raise_for_status()
        dictionary_response_payload = object_openfda_response.json()

        if integer_expected_total is None:
            integer_expected_total = int(dictionary_response_payload["meta"]["results"]["total"])

            if integer_expected_total <= 0:
                raise RuntimeError(f"No openFDA records were returned for {integer_year}.")

        list_page_records = dictionary_response_payload.get("results", [])

        if not list_page_records:
            raise RuntimeError(f"openFDA stopped before year {integer_year} was complete.")

        list_all_records.extend(list_page_records)
        integer_skip += len(list_page_records)

        if integer_skip < integer_expected_total:
            function_sleep(OPENFDA_REQUEST_PAUSE_SECONDS)

    if len(list_all_records) != integer_expected_total:
        raise RuntimeError(
            f"Year {integer_year} expected {integer_expected_total} records but "
            f"downloaded {len(list_all_records)}."
        )

    return list_all_records, integer_expected_total


def parse_openfda_date_values(series_date_values):
    return pandas.to_datetime(series_date_values, format="mixed", errors="coerce")


def prepare_downloaded_openfda_page(list_records, integer_expected_year):
    data_frame_year = pandas.json_normalize(list_records)

    data_frame_year = data_frame_year.rename(
        columns={
            "openfda.device_name": "device_name",
            ("openfda.medical_specialty_description"): "medical_specialty_description",
            "openfda.regulation_number": "regulation_number",
            "openfda.device_class": "device_class",
        }
    )

    list_retained_columns = [
        "cfres_id",
        "product_res_number",
        "res_event_number",
        "event_date_initiated",
        "event_date_posted",
        "event_date_terminated",
        "recall_status",
        "product_code",
        "product_description",
        "firm_fei_number",
        "recalling_firm",
        "city",
        "state",
        "reason_for_recall",
        "root_cause_description",
        "device_name",
        "medical_specialty_description",
        "regulation_number",
        "device_class",
    ]

    for string_column in list_retained_columns:
        if string_column not in data_frame_year.columns:
            data_frame_year[string_column] = pandas.NA

    data_frame_year = data_frame_year[list_retained_columns].copy()

    for string_date_column in ["event_date_initiated", "event_date_posted", "event_date_terminated"]:
        data_frame_year[string_date_column] = parse_openfda_date_values(data_frame_year[string_date_column])

    list_initiated_years = sorted(data_frame_year["event_date_initiated"].dropna().dt.year.unique().tolist())
    integer_unparsed_initiation_dates = int(data_frame_year["event_date_initiated"].isna().sum())

    if list_initiated_years != [int(integer_expected_year)]:
        raise RuntimeError(
            f"Downloaded records did not validate for year "
            f"{int(integer_expected_year)}. Observed parsed years: "
            f"{list_initiated_years or 'none'}; unparseable initiation "
            f"dates: {integer_unparsed_initiation_dates}."
        )

    return data_frame_year


def download_openfda_year_with_retries(
    integer_year,
    string_api_key=None,
    function_progress_callback=None,
    float_progress_fraction=0.02,
    function_request_get=None,
    function_sleep=None,
    integer_maximum_attempts=OPENFDA_YEAR_MAX_ATTEMPTS,
    float_retry_wait_seconds=OPENFDA_YEAR_RETRY_WAIT_SECONDS,
):
    if function_sleep is None:
        function_sleep = time.sleep

    integer_maximum_attempts = int(integer_maximum_attempts)
    float_retry_wait_seconds = float(float_retry_wait_seconds)

    if integer_maximum_attempts < 1:
        raise ValueError("maximum_attempts must be at least one.")

    if float_retry_wait_seconds < 0:
        raise ValueError("retry_wait_seconds cannot be negative.")

    exception_most_recent = None

    for integer_attempt_number in range(1, integer_maximum_attempts + 1):
        update_experiment_progress(
            function_progress_callback,
            float_progress_fraction,
            (
                f"Downloading openFDA records for {int(integer_year)} "
                f"(attempt {integer_attempt_number} of {integer_maximum_attempts})"
            ),
        )

        try:
            list_year_records, integer_expected_total = fetch_openfda_recall_year(
                integer_year=integer_year,
                string_api_key=string_api_key,
                function_request_get=function_request_get,
                function_sleep=function_sleep,
            )
            data_frame_year = prepare_downloaded_openfda_page(list_records=list_year_records, integer_expected_year=integer_year)

            if len(data_frame_year) != integer_expected_total:
                raise RuntimeError(
                    f"Year {integer_year} expected {integer_expected_total} records but "
                    f"prepared {len(data_frame_year)}."
                )

            return data_frame_year, integer_expected_total, integer_attempt_number

        except Exception as error:
            exception_most_recent = exception_error
            object_app_logger.warning(
                "openFDA year %s failed on attempt %s of %s: %s",
                int(integer_year),
                integer_attempt_number,
                integer_maximum_attempts,
                exception_error,
            )

            if integer_attempt_number == integer_maximum_attempts:
                break

            update_experiment_progress(
                function_progress_callback,
                float_progress_fraction,
                (
                    f"openFDA interrupted year {int(integer_year)}. "
                    f"Retrying in {float_retry_wait_seconds:g} seconds."
                ),
            )
            function_sleep(float_retry_wait_seconds)

    raise RuntimeError(
        f"openFDA year {int(integer_year)} failed after "
        f"{integer_maximum_attempts} attempts."
    ) from exception_most_recent


def download_openfda_year_range(
    integer_start_year,
    integer_end_year,
    string_api_key=None,
    function_progress_callback=None,
    function_request_get=None,
):
    string_active_api_key = resolve_openfda_api_key(
        string_api_key=string_api_key,
        function_progress_callback=function_progress_callback,
        function_request_get=function_request_get,
    )
    list_requested_years = list(range(int(integer_start_year), int(integer_end_year) + 1))
    list_data_frame_years = []
    list_year_audit_rows = []

    for integer_year_position, integer_year in enumerate(list_requested_years):
        float_progress_fraction = 0.02 + 0.18 * integer_year_position / len(list_requested_years)
        (data_frame_year, integer_expected_total, integer_attempts_used) = download_openfda_year_with_retries(
            integer_year=integer_year,
            string_api_key=string_active_api_key,
            function_progress_callback=function_progress_callback,
            float_progress_fraction=float_progress_fraction,
            function_request_get=function_request_get,
        )
        list_data_frame_years.append(data_frame_year)
        list_year_audit_rows.append({
            "year": int(integer_year),
            "expected_records": int(integer_expected_total),
            "downloaded_records": int(len(data_frame_year)),
            "download_attempts": int(integer_attempts_used),
            "complete": bool(len(data_frame_year) == integer_expected_total),
        })

    data_frame_year_audit = pandas.DataFrame(list_year_audit_rows)

    if (set(data_frame_year_audit["year"]) != set(list_requested_years) or not data_frame_year_audit["complete"].all()):
        raise RuntimeError("Not every requested year completed successfully.")

    data_frame_combined_records = pandas.concat(list_data_frame_years, ignore_index=True)

    return data_frame_combined_records, data_frame_year_audit


def prepare_pipeline_semantic_text(variable_value):
    if pandas.isna(variable_value):
        return ""

    string_text = unicodedata.normalize("NFKC", str(variable_value))
    string_text = re.sub(r"\s+", " ", string_text)

    return string_text.strip()


def generate_pipeline_semantic_text_id(string_text):
    return hashlib.sha256(string_text.encode("utf-8")).hexdigest()


def clean_pipeline_device_class(variable_value):
    if pandas.isna(variable_value) or not str(variable_value).strip():
        return "Missing / not available"

    string_class_code = str(variable_value).strip()
    dictionary_standard_labels = {"1": "Device Class I", "2": "Device Class II", "3": "Device Class III"}

    return dictionary_standard_labels.get(string_class_code, f"Other / unclassified code: {string_class_code}")


def build_unique_semantic_corpus(data_frame_full_records, list_manufacturer_terms, function_progress_callback=None):
    data_frame_semantic_source = data_frame_full_records.copy()

    data_frame_semantic_source["reason_for_recall"] = (data_frame_semantic_source["reason_for_recall"].fillna("").astype(str).str.strip())
    data_frame_semantic_source["reason_for_recall_original"] = data_frame_semantic_source["reason_for_recall"]

    series_prepared_reasons = data_frame_semantic_source["reason_for_recall"].str.lower()
    series_recalling_firms = data_frame_semantic_source["recalling_firm"].fillna("").astype(str)

    list_valid_manufacturer_terms = []

    for string_term in list_manufacturer_terms:
        if isinstance(string_term, str) and string_term.strip():
            list_valid_manufacturer_terms.append(string_term)

    for integer_term_position, string_manufacturer_term in enumerate(list_valid_manufacturer_terms):
        series_boolean_firm_mask = series_recalling_firms.str.contains(string_manufacturer_term, case=False, na=False, regex=False)

        series_prepared_reasons.loc[series_boolean_firm_mask] = (
            series_prepared_reasons.loc[series_boolean_firm_mask]
            .str.replace(string_manufacturer_term, "manufacturer", case=False, regex=False)
        )

        if integer_term_position % 250 == 0:
            update_experiment_progress(
                function_progress_callback,
                0.22 + 0.08 * integer_term_position / max(len(list_valid_manufacturer_terms), 1),
                "Masking manufacturer names",
            )

    series_prepared_reasons = series_prepared_reasons.str.replace("manufacturer inc.", "manufacturer", case=False, regex=False)

    for string_masked_word in PIPELINE_GENERIC_MASKED_WORDS:
        series_prepared_reasons = series_prepared_reasons.str.replace(rf"\b{re.escape(string_masked_word)}\b", "", case=False, regex=True)

    list_prepared_semantic_texts = []

    for string_prepared_reason in series_prepared_reasons:
        string_semantic_text = prepare_pipeline_semantic_text(string_prepared_reason)
        list_prepared_semantic_texts.append(string_semantic_text)

    data_frame_semantic_source["semantic_text"] = list_prepared_semantic_texts
    data_frame_semantic_source = data_frame_semantic_source[data_frame_semantic_source["semantic_text"].ne("")].copy()
    list_semantic_text_ids = []

    for string_semantic_text in data_frame_semantic_source["semantic_text"]:
        string_semantic_text_id = generate_pipeline_semantic_text_id(string_semantic_text)
        list_semantic_text_ids.append(string_semantic_text_id)

    data_frame_semantic_source["semantic_text_id"] = list_semantic_text_ids

    series_occurrence_counts = data_frame_semantic_source["semantic_text_id"].value_counts().rename("occurrence_count")
    data_frame_root_causes = data_frame_semantic_source[["semantic_text_id", "root_cause_description"]].copy()

    data_frame_root_causes["root_cause_clean"] = (
        data_frame_root_causes["root_cause_description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pandas.NA)
    )

    series_root_cause_counts = (
        data_frame_root_causes
        .groupby("semantic_text_id")["root_cause_clean"]
        .nunique(dropna=True)
        .rename("number_of_distinct_root_causes")
    )

    data_frame_unique_semantic_corpus = (
        data_frame_semantic_source[["semantic_text_id", "semantic_text"]]
        .drop_duplicates("semantic_text_id")
        .sort_values("semantic_text_id")
        .reset_index(drop=True)
        .merge(series_occurrence_counts, left_on="semantic_text_id", right_index=True, how="left")
        .merge(series_root_cause_counts, left_on="semantic_text_id", right_index=True, how="left")
    )
    data_frame_unique_semantic_corpus["number_of_distinct_root_causes"] = (
        data_frame_unique_semantic_corpus["number_of_distinct_root_causes"]
        .fillna(0)
        .astype(int)
    )
    data_frame_unique_semantic_corpus["embedding_row"] = numpy.arange(len(data_frame_unique_semantic_corpus), dtype=numpy.int32)

    list_identifier_columns = ["cfres_id", "product_res_number", "res_event_number"]
    data_frame_record_mapping = data_frame_semantic_source[list_identifier_columns + ["semantic_text_id", "semantic_text"]].copy()

    if len(data_frame_unique_semantic_corpus) < MINIMUM_UNIQUE_NARRATIVES:
        raise RuntimeError(
            "The selected years contain too few unique narratives "
            "for the fixed clustering configuration."
        )

    return (data_frame_semantic_source, data_frame_unique_semantic_corpus, data_frame_record_mapping)


def select_pipeline_torch_device(module_torch):
    if module_torch.cuda.is_available():
        return "cuda"

    if (hasattr(module_torch.backends, "mps") and module_torch.backends.mps.is_available()):
        return "mps"

    return "cpu"


def calculate_pipeline_token_lengths(list_texts, object_tokenizer, integer_batch_size=512):
    list_token_lengths = []

    for integer_start_position in range(0, len(list_texts), integer_batch_size):
        dictionary_encoded_batch = object_tokenizer(
            list_texts[integer_start_position: integer_start_position + integer_batch_size],
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_length=True,
            verbose=False,
        )
        list_token_lengths.extend(dictionary_encoded_batch["length"])

    return numpy.asarray(list_token_lengths, dtype=numpy.int32)


def calculate_pipeline_silhouette(array_coordinates, array_topic_assignments, string_metric):
    from sklearn.metrics import silhouette_score

    array_topic_assignments = numpy.asarray(array_topic_assignments)
    array_boolean_assigned_mask = array_topic_assignments != -1
    array_assigned_topics = numpy.unique(array_topic_assignments[array_boolean_assigned_mask])
    integer_assigned_count = int(array_boolean_assigned_mask.sum())

    if len(array_assigned_topics) < 2 or integer_assigned_count <= len(array_assigned_topics):
        return None

    return float(
        silhouette_score(
            array_coordinates[array_boolean_assigned_mask],
            array_topic_assignments[array_boolean_assigned_mask],
            metric=string_metric,
            sample_size=min(3_000, integer_assigned_count),
            random_state=42,
        )
    )


def build_automatic_topic_labels(model_topic, set_topic_ids):
    set_normalised_topic_ids = set()

    for variable_topic_id in set_topic_ids:
        set_normalised_topic_ids.add(int(variable_topic_id))

    set_topic_ids = set_normalised_topic_ids
    dictionary_automatic_labels = {}

    if -1 in set_topic_ids:
        dictionary_automatic_labels[-1] = "Complex / unassigned narratives"

    for integer_topic_id in sorted(set_topic_ids):
        if integer_topic_id == -1:
            continue

        list_topic_terms = model_topic.get_topic(integer_topic_id) or []
        list_top_terms = []

        for string_term, float_score in list_topic_terms[:4]:
            string_readable_term = str(string_term).replace("_", " ")
            list_top_terms.append(string_readable_term)

        if list_top_terms:
            dictionary_automatic_labels[integer_topic_id] = "Keyword-derived: " + ", ".join(list_top_terms)
        else:
            dictionary_automatic_labels[integer_topic_id] = f"Keyword-derived topic {integer_topic_id}"

    return dictionary_automatic_labels


def load_pipeline_embedding_model(
    class_sentence_transformer,
    string_selected_device,
    string_huggingface_token=None,
    function_progress_callback=None,
):
    string_cleaned_token = str(string_huggingface_token).strip() if string_huggingface_token else ""

    try:
        return class_sentence_transformer(EXPERIMENT_MODEL_NAME, device=string_selected_device, local_files_only=True)
    except Exception:
        pass

    if string_cleaned_token:
        try:
            return class_sentence_transformer(EXPERIMENT_MODEL_NAME, device=string_selected_device, token=string_cleaned_token)
        except Exception:
            object_app_logger.warning(
                "The supplied Hugging Face token was not accepted. "
                "Continuing without a token."
            )
            update_experiment_progress(
                function_progress_callback,
                0.32,
                (
                    "Hugging Face token not accepted. "
                    "Continuing without a token."
                ),
            )

    return class_sentence_transformer(EXPERIMENT_MODEL_NAME, device=string_selected_device)


def fit_fixed_bertopic_pipeline(
    data_frame_unique_semantic_corpus,
    float_outlier_threshold,
    string_huggingface_token=None,
    function_progress_callback=None,
):
    string_cache_root = Path(tempfile.gettempdir()) / "fda_recall_ml_cache"
    string_numba_cache = string_cache_root / "numba"
    string_matplotlib_cache = string_cache_root / "matplotlib"
    string_numba_cache.mkdir(parents=True, exist_ok=True)
    string_matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(string_numba_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(string_matplotlib_cache))

    import torch
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import (CountVectorizer, ENGLISH_STOP_WORDS)
    from umap import UMAP

    random.seed(EXPERIMENT_RANDOM_SEED)
    numpy.random.seed(EXPERIMENT_RANDOM_SEED)
    torch.manual_seed(EXPERIMENT_RANDOM_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        torch.mps.manual_seed(EXPERIMENT_RANDOM_SEED)

    string_selected_device = select_pipeline_torch_device(torch)
    update_experiment_progress(function_progress_callback, 0.32, f"Loading MPNet on {string_selected_device}")
    model_embedding = load_pipeline_embedding_model(
        class_sentence_transformer=SentenceTransformer,
        string_selected_device=string_selected_device,
        string_huggingface_token=string_huggingface_token,
        function_progress_callback=function_progress_callback,
    )

    list_documents = data_frame_unique_semantic_corpus["semantic_text"].astype(str).tolist()
    array_token_lengths = calculate_pipeline_token_lengths(list_texts=list_documents, object_tokenizer=model_embedding.tokenizer)
    integer_maximum_sequence_length = int(model_embedding.max_seq_length)
    data_frame_unique_semantic_corpus = data_frame_unique_semantic_corpus.copy()
    data_frame_unique_semantic_corpus["token_length"] = array_token_lengths
    data_frame_unique_semantic_corpus["will_be_truncated"] = array_token_lengths > integer_maximum_sequence_length

    update_experiment_progress(function_progress_callback, 0.38, "Encoding unique narratives with MPNet")
    array_sentence_embeddings = model_embedding.encode(
        list_documents,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
    ).astype(numpy.float32, copy=False)

    if (
        array_sentence_embeddings.shape[0] != len(data_frame_unique_semantic_corpus)
        or array_sentence_embeddings.shape[1] != 768
        or not numpy.isfinite(array_sentence_embeddings).all()
    ):
        raise RuntimeError("The MPNet embedding matrix failed validation.")

    if not numpy.allclose(numpy.linalg.norm(array_sentence_embeddings, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("The MPNet embeddings are not normalized.")

    set_generic_recall_words = {
        "recall",
        "recalled",
        "recalling",
        "device",
        "devices",
        "product",
        "products",
        "unit",
        "units",
        "firm",
        "company",
        "companies",
        "affected",
        "certain",
        "potential",
        "potentially",
        "may",
        "could",
        "issue",
        "issues",
    }
    list_topic_stop_words = sorted(set(ENGLISH_STOP_WORDS).union(set_generic_recall_words))
    model_umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    model_hdbscan = HDBSCAN(min_cluster_size=40, min_samples=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    model_vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list_topic_stop_words,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
    )
    model_ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    model_topic = BERTopic(
        embedding_model=model_embedding,
        umap_model=model_umap,
        hdbscan_model=model_hdbscan,
        vectorizer_model=model_vectorizer,
        ctfidf_model=model_ctfidf,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )

    update_experiment_progress(function_progress_callback, 0.56, "Fitting fixed UMAP, HDBSCAN, and BERTopic stages")
    array_original_topics, array_probabilities = model_topic.fit_transform(list_documents, embeddings=array_sentence_embeddings)
    array_original_topics = numpy.asarray(array_original_topics, dtype=numpy.int32)
    array_substantive_topic_ids = numpy.unique(array_original_topics[array_original_topics != -1])

    if len(array_substantive_topic_ids) == 0:
        raise RuntimeError("The fixed model produced no substantive topics.")

    update_experiment_progress(function_progress_callback, 0.68, "Conservatively reassigning eligible outliers")

    if numpy.any(array_original_topics == -1):
        array_final_topics = numpy.asarray(
            model_topic.reduce_outliers(
                list_documents,
                array_original_topics,
                strategy="embeddings",
                embeddings=array_sentence_embeddings,
                threshold=float(float_outlier_threshold),
            ),
            dtype=numpy.int32,
        )
    else:
        array_final_topics = array_original_topics.copy()

    model_representation = KeyBERTInspired(top_n_words=10, nr_repr_docs=5, nr_samples=500, nr_candidate_words=100, random_state=42)
    model_topic.update_topics(
        list_documents,
        topics=array_final_topics,
        vectorizer_model=model_vectorizer,
        ctfidf_model=model_ctfidf,
        representation_model=model_representation,
        top_n_words=10,
    )

    set_final_topic_ids = set()

    for variable_topic_id in numpy.unique(array_final_topics):
        set_final_topic_ids.add(int(variable_topic_id))

    dictionary_automatic_topic_labels = build_automatic_topic_labels(model_topic=model_topic, set_topic_ids=set_final_topic_ids)
    model_topic.set_topic_labels(dictionary_automatic_topic_labels)
    data_frame_topic_information = model_topic.get_topic_info().copy()

    list_topic_keyword_rows = []

    for integer_topic_id in sorted(set_final_topic_ids):
        for integer_rank, (string_term, float_score) in enumerate(model_topic.get_topic(integer_topic_id) or [], start=1):
            list_topic_keyword_rows.append({
                "topic": int(integer_topic_id),
                "rank": int(integer_rank),
                "term": str(string_term),
                "score": float(float_score),
            })

    data_frame_topic_keywords = pandas.DataFrame(list_topic_keyword_rows)
    list_representative_rows = []

    for integer_topic_id, list_representative_documents_for_topic in (model_topic.get_representative_docs() or {}).items():
        if int(integer_topic_id) == -1:
            continue

        for integer_rank, string_representative_document in enumerate(list_representative_documents_for_topic, start=1):
            list_representative_rows.append({
                "topic": int(integer_topic_id),
                "representative_rank": int(integer_rank),
                "representative_document": str(string_representative_document),
            })

    data_frame_representative_document_table = pandas.DataFrame(
        list_representative_rows,
        columns=["topic", "representative_rank", "representative_document"],
    )
    dictionary_topic_name_lookup = data_frame_topic_information.set_index("Topic")["Name"].to_dict()
    data_frame_document_topics = data_frame_unique_semantic_corpus.copy()
    data_frame_document_topics["bertopic_Topic"] = array_final_topics
    data_frame_document_topics["bertopic_Name"] = (
        data_frame_document_topics["bertopic_Topic"]
        .map(dictionary_topic_name_lookup)
        .fillna("Unlabelled topic")
    )
    data_frame_document_topics["bertopic_CustomName"] = (
        data_frame_document_topics["bertopic_Topic"]
        .map(dictionary_automatic_topic_labels)
        .fillna("Unlabelled topic")
    )
    array_reduced_embeddings_5d = numpy.asarray(model_topic.umap_model.embedding_, dtype=numpy.float32)
    dictionary_silhouette_metrics = {
        "original_embedding_cosine": calculate_pipeline_silhouette(
            array_sentence_embeddings,
            array_original_topics,
            string_metric="cosine",
        ),
        "final_embedding_cosine": calculate_pipeline_silhouette(array_sentence_embeddings, array_final_topics, string_metric="cosine"),
        "original_reduced_euclidean": calculate_pipeline_silhouette(
            array_reduced_embeddings_5d,
            array_original_topics,
            string_metric="euclidean",
        ),
        "final_reduced_euclidean": calculate_pipeline_silhouette(
            array_reduced_embeddings_5d,
            array_final_topics,
            string_metric="euclidean",
        ),
    }

    update_experiment_progress(function_progress_callback, 0.76, "Creating the two-dimensional semantic map")
    model_visualisation_umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    array_sentence_embeddings_2d = (model_visualisation_umap.fit_transform(array_sentence_embeddings).astype(numpy.float32, copy=False))

    return {
        "embedding_model": model_embedding,
        "device": string_selected_device,
        "documents": list_documents,
        "unique_semantic_corpus": data_frame_unique_semantic_corpus,
        "sentence_embeddings": array_sentence_embeddings,
        "sentence_embeddings_2d": array_sentence_embeddings_2d,
        "original_topics": array_original_topics,
        "final_topics": array_final_topics,
        "topic_model": model_topic,
        "topic_information": data_frame_topic_information,
        "topic_keywords": data_frame_topic_keywords,
        "representative_documents": data_frame_representative_document_table,
        "document_topics": data_frame_document_topics,
        "automatic_topic_labels": dictionary_automatic_topic_labels,
        "silhouette_metrics": dictionary_silhouette_metrics,
        "maximum_sequence_length": integer_maximum_sequence_length,
        "probabilities": array_probabilities,
    }


def build_pipeline_dashboard_records(data_frame_semantic_source, data_frame_document_topics, dictionary_automatic_topic_labels):
    data_frame_topic_lookup = data_frame_document_topics[
        ["semantic_text_id", "bertopic_Topic", "bertopic_Name"]
    ].drop_duplicates("semantic_text_id")
    data_frame_dashboard_records = data_frame_semantic_source.merge(
        data_frame_topic_lookup,
        on="semantic_text_id",
        how="left",
        validate="many_to_one",
    )
    data_frame_dashboard_records["topic_label"] = (
        data_frame_dashboard_records["bertopic_Topic"]
        .map(dictionary_automatic_topic_labels)
        .fillna("Unlabelled topic")
    )
    data_frame_dashboard_records["occurrence_count"] = (
        data_frame_dashboard_records["semantic_text_id"]
        .map(data_frame_dashboard_records["semantic_text_id"].value_counts())
        .astype(int)
    )
    data_frame_dashboard_records["event_date_initiated"] = pandas.to_datetime(
        data_frame_dashboard_records["event_date_initiated"],
        errors="coerce",
    )
    data_frame_dashboard_records["year_initiated"] = (data_frame_dashboard_records["event_date_initiated"].dt.year.astype("Int64"))
    data_frame_dashboard_records["recalling_firm_clean"] = (
        data_frame_dashboard_records["recalling_firm"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .fillna("Missing / not available")
    )
    list_clean_device_classes = []

    for variable_device_class in data_frame_dashboard_records["device_class"]:
        string_clean_device_class = clean_pipeline_device_class(variable_device_class)
        list_clean_device_classes.append(string_clean_device_class)

    data_frame_dashboard_records["device_class_clean"] = list_clean_device_classes
    data_frame_dashboard_records["is_complex_narrative"] = data_frame_dashboard_records["bertopic_Topic"] == -1

    list_dashboard_columns = [
        "cfres_id",
        "product_res_number",
        "res_event_number",
        "event_date_initiated",
        "year_initiated",
        "reason_for_recall",
        "semantic_text_id",
        "semantic_text",
        "occurrence_count",
        "bertopic_Topic",
        "bertopic_Name",
        "topic_label",
        "recalling_firm",
        "recalling_firm_clean",
        "firm_fei_number",
        "device_name",
        "device_class",
        "device_class_clean",
        "medical_specialty_description",
        "product_code",
        "product_description",
        "state",
        "root_cause_description",
        "recall_status",
        "is_complex_narrative",
    ]

    for string_column in list_dashboard_columns:
        if string_column not in data_frame_dashboard_records.columns:
            data_frame_dashboard_records[string_column] = pandas.NA

    return data_frame_dashboard_records[list_dashboard_columns].copy()


def calculate_experiment_cluster_summary(data_frame_dashboard_records, dictionary_topic_label_lookup):
    data_frame_cluster_summary = (
        data_frame_dashboard_records
        .groupby("bertopic_Topic", as_index=False)
        .agg(
            full_record_count=("cfres_id", "size"),
            unique_narrative_count=("semantic_text_id", "nunique"),
            unique_recall_event_count=("res_event_number", "nunique"),
        )
        .rename(columns={"bertopic_Topic": "topic"})
    )
    data_frame_cluster_summary["topic_label"] = (
        data_frame_cluster_summary["topic"]
        .map(dictionary_topic_label_lookup)
        .fillna("Unlabelled topic")
    )
    data_frame_cluster_summary["full_record_percentage"] = (
        100
        * data_frame_cluster_summary["full_record_count"]
        / data_frame_cluster_summary["full_record_count"].sum()
    )
    data_frame_cluster_summary["unique_narrative_percentage"] = (
        100
        * data_frame_cluster_summary["unique_narrative_count"]
        / data_frame_cluster_summary["unique_narrative_count"].sum()
    )

    return data_frame_cluster_summary.sort_values("full_record_count", ascending=False).reset_index(drop=True)


def create_experiment_static_figures(
    data_frame_dashboard_records,
    data_frame_document_topics,
    array_sentence_embeddings_2d,
    string_output_directory,
):
    string_cache_root = Path(tempfile.gettempdir()) / "fda_recall_ml_cache"
    string_matplotlib_cache = string_cache_root / "matplotlib"
    string_matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(string_matplotlib_cache))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    string_output_directory = Path(string_output_directory)
    string_output_directory.mkdir(parents=True, exist_ok=True)
    list_figure_paths = []
    string_colour_primary = "#3B6F8F"
    string_colour_complex = "#C94C5C"
    string_colour_increase = "#2A9D8F"
    string_colour_decrease = "#C94C5C"
    string_colour_reference = "#555555"
    list_topic_colours = list(plt.get_cmap("tab20").colors)

    def wrap_label(variable_value, integer_width=46):
        return textwrap.fill(str(variable_value), width=integer_width)

    def save_figure(figure_to_save, string_file_stem):
        string_png_path = string_output_directory / f"{string_file_stem}.png"
        string_pdf_path = string_output_directory / f"{string_file_stem}.pdf"
        figure_to_save.savefig(string_png_path, dpi=220, bbox_inches="tight", facecolor="white")
        figure_to_save.savefig(string_pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure_to_save)
        list_figure_paths.append(string_png_path)

    data_frame_substantive_records = data_frame_dashboard_records[data_frame_dashboard_records["bertopic_Topic"] != -1]
    data_frame_prevalence_table = (
        data_frame_substantive_records
        .groupby(["bertopic_Topic", "topic_label"], as_index=False)
        .size()
        .rename(columns={"size": "record_count"})
    )
    data_frame_prevalence_table["record_percentage"] = (
        100
        * data_frame_prevalence_table["record_count"]
        / max(len(data_frame_substantive_records), 1)
    )
    data_frame_prevalence_table = (data_frame_prevalence_table.nlargest(15, "record_count").sort_values("record_percentage").copy())
    list_prevalence_display_labels = []

    for _, series_prevalence_row in data_frame_prevalence_table.iterrows():
        string_display_label = (
            f"T{int(series_prevalence_row['bertopic_Topic'])}: "
            f"{series_prevalence_row['topic_label']}"
        )
        list_prevalence_display_labels.append(wrap_label(string_display_label))

    data_frame_prevalence_table["display_label"] = list_prevalence_display_labels
    figure_plot, axis_plot = plt.subplots(figsize=(11, 8))
    object_bars = axis_plot.barh(
        data_frame_prevalence_table["display_label"],
        data_frame_prevalence_table["record_percentage"],
        color=string_colour_primary,
    )
    list_prevalence_percentage_labels = []

    for float_percentage in data_frame_prevalence_table["record_percentage"]:
        list_prevalence_percentage_labels.append(f"{float_percentage:.1f}%")

    axis_plot.bar_label(object_bars, labels=list_prevalence_percentage_labels, padding=3, fontsize=8)
    axis_plot.set_title("Most Prevalent Discovered Failure Patterns")
    axis_plot.set_xlabel("Share of substantively assigned recall records")
    axis_plot.set_ylabel("Failure pattern")
    axis_plot.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis_plot.grid(axis="x", alpha=0.25)
    axis_plot.set_axisbelow(True)
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_01_topic_prevalence")

    list_top_temporal_topics = (data_frame_prevalence_table.nlargest(6, "record_count")["bertopic_Topic"].astype(int).tolist())
    series_yearly_totals = data_frame_dashboard_records.groupby("year_initiated").size().rename("year_total")
    data_frame_yearly_topics = (
        data_frame_dashboard_records[data_frame_dashboard_records["bertopic_Topic"].isin(list_top_temporal_topics)]
        .groupby(["year_initiated", "bertopic_Topic", "topic_label"])
        .size()
        .rename("topic_count")
        .reset_index()
        .merge(series_yearly_totals, left_on="year_initiated", right_index=True, how="left")
    )
    data_frame_yearly_topics["topic_percentage"] = (100 * data_frame_yearly_topics["topic_count"] / data_frame_yearly_topics["year_total"])
    figure_plot, axis_plot = plt.subplots(figsize=(12, 7))

    for integer_topic_position, integer_topic_id in enumerate(list_top_temporal_topics):
        data_frame_topic_data = data_frame_yearly_topics[
            data_frame_yearly_topics["bertopic_Topic"] == integer_topic_id
        ].sort_values("year_initiated")

        if data_frame_topic_data.empty:
            continue

        axis_plot.plot(
            data_frame_topic_data["year_initiated"],
            data_frame_topic_data["topic_percentage"],
            marker="o",
            linewidth=2,
            color=list_topic_colours[integer_topic_position % len(list_topic_colours)],
            label=wrap_label(f"T{integer_topic_id}: {data_frame_topic_data['topic_label'].iloc[0]}", integer_width=42),
        )

    axis_plot.set_title("Annual Prevalence of Major Failure Patterns")
    axis_plot.set_xlabel("Recall initiation year")
    axis_plot.set_ylabel("Share of recall records within each year")
    axis_plot.set_ylim(bottom=0)
    axis_plot.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis_plot.grid(alpha=0.25)
    axis_plot.set_axisbelow(True)
    axis_plot.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_02_temporal_prevalence")

    list_all_years = sorted(data_frame_dashboard_records["year_initiated"].dropna().astype(int).unique().tolist())
    integer_number_of_period_years = min(3, max(1, len(list_all_years) // 2))
    list_early_years = list_all_years[:integer_number_of_period_years]
    list_recent_years = list_all_years[-integer_number_of_period_years:]
    list_substantive_topic_ids = sorted(data_frame_substantive_records["bertopic_Topic"].astype(int).unique().tolist())
    dictionary_topic_label_lookup = (
        data_frame_substantive_records
        .drop_duplicates("bertopic_Topic")
        .set_index("bertopic_Topic")["topic_label"]
        .to_dict()
    )

    pandas_index_topic_year_grid = pandas.MultiIndex.from_product(
        [list_all_years, list_substantive_topic_ids],
        names=["year_initiated", "bertopic_Topic"],
    ).to_frame(index=False)
    data_frame_topic_year_counts = (
        data_frame_substantive_records
        .groupby(["year_initiated", "bertopic_Topic"])
        .size()
        .rename("topic_count")
        .reset_index()
    )
    data_frame_topic_year_trends = (
        pandas_index_topic_year_grid
        .merge(data_frame_topic_year_counts, on=["year_initiated", "bertopic_Topic"], how="left")
        .merge(series_yearly_totals.rename_axis("year_initiated").reset_index(), on="year_initiated", how="left")
    )
    data_frame_topic_year_trends["topic_count"] = data_frame_topic_year_trends["topic_count"].fillna(0)
    data_frame_topic_year_trends["topic_percentage"] = (
        100
        * data_frame_topic_year_trends["topic_count"]
        / data_frame_topic_year_trends["year_total"]
    )

    list_temporal_change_rows = []

    for integer_topic_id in list_substantive_topic_ids:
        data_frame_topic_trend = data_frame_topic_year_trends[data_frame_topic_year_trends["bertopic_Topic"] == integer_topic_id]
        float_early_mean = data_frame_topic_trend.loc[
            data_frame_topic_trend["year_initiated"].isin(list_early_years),
            "topic_percentage",
        ].mean()
        float_recent_mean = data_frame_topic_trend.loc[
            data_frame_topic_trend["year_initiated"].isin(list_recent_years),
            "topic_percentage",
        ].mean()
        list_temporal_change_rows.append({
            "topic": int(integer_topic_id),
            "topic_label": dictionary_topic_label_lookup[integer_topic_id],
            "total_count": int(data_frame_topic_trend["topic_count"].sum()),
            "change_percentage_points": float(float_recent_mean - float_early_mean),
        })

    data_frame_temporal_change_summary = pandas.DataFrame(list_temporal_change_rows)
    data_frame_temporal_change_candidates = data_frame_temporal_change_summary[
        data_frame_temporal_change_summary["total_count"] >= 100
    ].copy()
    data_frame_temporal_change_selection = (
        pandas.concat(
            [
                data_frame_temporal_change_candidates.nsmallest(7, "change_percentage_points"),
                data_frame_temporal_change_candidates.nlargest(7, "change_percentage_points"),
            ],
            ignore_index=True,
        )
        .drop_duplicates("topic")
        .sort_values("change_percentage_points")
        .copy()
    )
    figure_plot, axis_plot = plt.subplots(figsize=(12, 8))

    if data_frame_temporal_change_selection.empty:
        axis_plot.text(0.5, 0.5, "No topic met the 100-record support threshold.", ha="center", va="center", transform=axis_plot.transAxes)
        axis_plot.set_axis_off()
    else:
        list_temporal_change_display_labels = []

        for _, series_temporal_change_row in data_frame_temporal_change_selection.iterrows():
            string_display_label = (
                f"T{int(series_temporal_change_row['topic'])}: "
                f"{series_temporal_change_row['topic_label']}"
            )
            list_temporal_change_display_labels.append(wrap_label(string_display_label, integer_width=50))

        data_frame_temporal_change_selection["display_label"] = list_temporal_change_display_labels
        array_change_colours = numpy.where(
            data_frame_temporal_change_selection["change_percentage_points"] >= 0,
            string_colour_increase,
            string_colour_decrease,
        )
        axis_plot.barh(
            data_frame_temporal_change_selection["display_label"],
            data_frame_temporal_change_selection["change_percentage_points"],
            color=array_change_colours,
        )
        axis_plot.axvline(0, color=string_colour_reference, linewidth=1)
        axis_plot.set_xlabel(
            f"{list_recent_years[0]}-{list_recent_years[-1]} mean minus "
            f"{list_early_years[0]}-{list_early_years[-1]} mean "
            "(percentage points)"
        )
        axis_plot.set_ylabel("Failure pattern")
        axis_plot.grid(axis="x", alpha=0.25)
        axis_plot.set_axisbelow(True)

    axis_plot.set_title("Largest Changes in Failure-Pattern Prevalence")
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_03_temporal_change")

    data_frame_device_source = data_frame_dashboard_records[
        data_frame_dashboard_records["device_class_clean"].isin(["Device Class I", "Device Class II", "Device Class III"])
        & data_frame_dashboard_records["bertopic_Topic"].isin(list_top_temporal_topics)
    ].copy()
    data_frame_device_counts = (
        data_frame_device_source
        .groupby(["device_class_clean", "bertopic_Topic", "topic_label"])
        .size()
        .rename("topic_count")
        .reset_index()
    )
    data_frame_device_counts["class_total"] = (
        data_frame_device_counts["device_class_clean"]
        .map(data_frame_device_source["device_class_clean"].value_counts())
    )
    data_frame_device_counts["topic_percentage"] = (100 * data_frame_device_counts["topic_count"] / data_frame_device_counts["class_total"])
    list_device_topic_display_labels = []

    for _, series_device_count_row in data_frame_device_counts.iterrows():
        string_display_label = (
            f"T{int(series_device_count_row['bertopic_Topic'])}: "
            f"{series_device_count_row['topic_label']}"
        )
        list_device_topic_display_labels.append(wrap_label(string_display_label, integer_width=34))

    data_frame_device_counts["topic_display"] = list_device_topic_display_labels
    data_frame_device_heatmap = (
        data_frame_device_counts
        .pivot(index="topic_display", columns="device_class_clean", values="topic_percentage")
        .fillna(0)
    )
    figure_plot, axis_plot = plt.subplots(figsize=(10, 7))
    image_heatmap = axis_plot.imshow(data_frame_device_heatmap.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0)
    axis_plot.set_xticks(numpy.arange(len(data_frame_device_heatmap.columns)))
    axis_plot.set_xticklabels(data_frame_device_heatmap.columns)
    axis_plot.set_yticks(numpy.arange(len(data_frame_device_heatmap.index)))
    axis_plot.set_yticklabels(data_frame_device_heatmap.index, fontsize=8)
    axis_plot.set_title("Failure-Pattern Prevalence by Device Class")
    axis_plot.set_xlabel("Device regulatory class")
    axis_plot.set_ylabel("Failure pattern")
    figure_plot.colorbar(image_heatmap, ax=axis_plot).set_label("Share of class records (%)")
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_04_topics_by_device_class")

    list_top_firms = data_frame_dashboard_records["recalling_firm_clean"].value_counts().head(10).index.tolist()
    data_frame_firm_source = data_frame_dashboard_records[
        data_frame_dashboard_records["recalling_firm_clean"].isin(list_top_firms)
        & data_frame_dashboard_records["bertopic_Topic"].isin(list_top_temporal_topics)
    ].copy()
    data_frame_firm_counts = (
        data_frame_firm_source
        .groupby(["recalling_firm_clean", "bertopic_Topic", "topic_label"])
        .size()
        .rename("topic_count")
        .reset_index()
    )
    data_frame_firm_counts["firm_total"] = (
        data_frame_firm_counts["recalling_firm_clean"]
        .map(data_frame_dashboard_records["recalling_firm_clean"].value_counts())
    )
    data_frame_firm_counts["topic_percentage"] = (100 * data_frame_firm_counts["topic_count"] / data_frame_firm_counts["firm_total"])
    list_firm_topic_display_labels = []

    for _, series_firm_count_row in data_frame_firm_counts.iterrows():
        string_display_label = (
            f"T{int(series_firm_count_row['bertopic_Topic'])}: "
            f"{series_firm_count_row['topic_label']}"
        )
        list_firm_topic_display_labels.append(wrap_label(string_display_label, integer_width=25))

    data_frame_firm_counts["topic_display"] = list_firm_topic_display_labels
    data_frame_firm_heatmap = (
        data_frame_firm_counts
        .pivot(index="recalling_firm_clean", columns="topic_display", values="topic_percentage")
        .reindex(index=list_top_firms)
        .fillna(0)
    )
    figure_plot, axis_plot = plt.subplots(figsize=(15, 8))
    image_heatmap = axis_plot.imshow(data_frame_firm_heatmap.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0)
    axis_plot.set_xticks(numpy.arange(len(data_frame_firm_heatmap.columns)))
    axis_plot.set_xticklabels(data_frame_firm_heatmap.columns, rotation=55, ha="right", fontsize=8)
    axis_plot.set_yticks(numpy.arange(len(data_frame_firm_heatmap.index)))
    list_firm_axis_labels = []

    for string_firm_name in data_frame_firm_heatmap.index:
        list_firm_axis_labels.append(wrap_label(string_firm_name, integer_width=32))

    axis_plot.set_yticklabels(list_firm_axis_labels, fontsize=8)
    axis_plot.set_title("Failure-Pattern Profiles of Major Recalling Firms")
    axis_plot.set_xlabel("Failure pattern")
    axis_plot.set_ylabel("Recalling firm")
    figure_plot.colorbar(image_heatmap, ax=axis_plot).set_label("Share of firm records (%)")
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_05_firm_topic_profiles")

    data_frame_complex_records = data_frame_dashboard_records[data_frame_dashboard_records["bertopic_Topic"] == -1]
    data_frame_complex_yearly = (
        data_frame_complex_records
        .groupby("year_initiated")
        .size()
        .rename("complex_count")
        .to_frame()
        .join(series_yearly_totals, how="right")
        .fillna({"complex_count": 0})
        .reset_index()
    )
    data_frame_complex_yearly["complex_percentage"] = (
        100
        * data_frame_complex_yearly["complex_count"]
        / data_frame_complex_yearly["year_total"]
    )
    figure_plot, axis_plot = plt.subplots(figsize=(10, 5))
    axis_plot.plot(
        data_frame_complex_yearly["year_initiated"],
        data_frame_complex_yearly["complex_percentage"],
        marker="o",
        linewidth=2.2,
        color=string_colour_complex,
    )
    axis_plot.set_title("Complex or Unassigned Narratives Over Time")
    axis_plot.set_xlabel("Recall initiation year")
    axis_plot.set_ylabel("Share of recall records within each year")
    axis_plot.set_ylim(bottom=0)
    axis_plot.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis_plot.grid(alpha=0.25)
    axis_plot.set_axisbelow(True)
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_06_complex_narratives_temporal")

    data_frame_event_source = data_frame_dashboard_records[
        data_frame_dashboard_records["res_event_number"].notna()
        & data_frame_dashboard_records["res_event_number"].ne("")
    ].copy()
    series_event_year_totals = (data_frame_event_source.groupby("year_initiated")["res_event_number"].nunique().rename("year_event_total"))
    data_frame_event_topic_year_counts = (
        data_frame_event_source[data_frame_event_source["bertopic_Topic"].isin(list_substantive_topic_ids)]
        .groupby(["year_initiated", "bertopic_Topic"])["res_event_number"]
        .nunique()
        .rename("topic_event_count")
        .reset_index()
    )
    data_frame_event_topic_year_trends = (
        pandas_index_topic_year_grid
        .merge(data_frame_event_topic_year_counts, on=["year_initiated", "bertopic_Topic"], how="left")
        .merge(series_event_year_totals.rename_axis("year_initiated").reset_index(), on="year_initiated", how="left")
    )
    data_frame_event_topic_year_trends["topic_event_count"] = (data_frame_event_topic_year_trends["topic_event_count"].fillna(0))
    data_frame_event_topic_year_trends["topic_percentage"] = (
        100
        * data_frame_event_topic_year_trends["topic_event_count"]
        / data_frame_event_topic_year_trends["year_event_total"]
    )

    list_sensitivity_rows = []

    for integer_topic_id in list_substantive_topic_ids:
        data_frame_record_trend = data_frame_topic_year_trends[data_frame_topic_year_trends["bertopic_Topic"] == integer_topic_id]
        data_frame_event_trend = data_frame_event_topic_year_trends[
            data_frame_event_topic_year_trends["bertopic_Topic"] == integer_topic_id
        ]
        float_record_change = (
            data_frame_record_trend.loc[data_frame_record_trend["year_initiated"].isin(list_recent_years), "topic_percentage"].mean()
            - data_frame_record_trend.loc[data_frame_record_trend["year_initiated"].isin(list_early_years), "topic_percentage"].mean()
        )
        float_event_change = (
            data_frame_event_trend.loc[data_frame_event_trend["year_initiated"].isin(list_recent_years), "topic_percentage"].mean()
            - data_frame_event_trend.loc[data_frame_event_trend["year_initiated"].isin(list_early_years), "topic_percentage"].mean()
        )
        list_sensitivity_rows.append({
            "topic": int(integer_topic_id),
            "topic_label": dictionary_topic_label_lookup[integer_topic_id],
            "record_total_count": int(data_frame_record_trend["topic_count"].sum()),
            "event_total_count": int(data_frame_event_trend["topic_event_count"].sum()),
            "change_difference_event_minus_record": float(float_event_change - float_record_change),
        })

    data_frame_sensitivity_summary = pandas.DataFrame(list_sensitivity_rows)
    data_frame_sensitivity_candidates = data_frame_sensitivity_summary[
        (data_frame_sensitivity_summary["record_total_count"] >= 100)
        & (data_frame_sensitivity_summary["event_total_count"] >= 20)
    ].copy()
    figure_plot, axis_plot = plt.subplots(figsize=(10, 6))

    if data_frame_sensitivity_candidates.empty:
        axis_plot.text(
            0.5,
            0.5,
            "No topic met the record and event support thresholds.",
            ha="center",
            va="center",
            transform=axis_plot.transAxes,
        )
        axis_plot.set_axis_off()
        string_sensitivity_title = "Record- versus Event-Weighted Prevalence"
    else:
        data_frame_sensitivity_candidates["absolute_change_difference"] = (
            data_frame_sensitivity_candidates["change_difference_event_minus_record"].abs()
        )
        series_sensitivity_topic = data_frame_sensitivity_candidates.nlargest(1, "absolute_change_difference").iloc[0]
        integer_sensitivity_topic_id = int(series_sensitivity_topic["topic"])
        string_sensitivity_topic_label = series_sensitivity_topic["topic_label"]
        data_frame_record_weighted_trend = data_frame_topic_year_trends[
            data_frame_topic_year_trends["bertopic_Topic"]
            == integer_sensitivity_topic_id
        ].sort_values("year_initiated")
        data_frame_event_weighted_trend = data_frame_event_topic_year_trends[
            data_frame_event_topic_year_trends["bertopic_Topic"]
            == integer_sensitivity_topic_id
        ].sort_values("year_initiated")
        axis_plot.plot(
            data_frame_record_weighted_trend["year_initiated"],
            data_frame_record_weighted_trend["topic_percentage"],
            marker="o",
            linewidth=2,
            color=string_colour_primary,
            label="Product recall records",
        )
        axis_plot.plot(
            data_frame_event_weighted_trend["year_initiated"],
            data_frame_event_weighted_trend["topic_percentage"],
            marker="s",
            linewidth=2,
            color=string_colour_increase,
            label="Unique recall events containing the topic",
        )
        axis_plot.set_xlabel("Recall initiation year")
        axis_plot.set_ylabel("Topic prevalence within each denominator")
        axis_plot.set_ylim(bottom=0)
        axis_plot.set_xticks(list_all_years)
        axis_plot.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis_plot.legend(frameon=False)
        axis_plot.grid(alpha=0.25)
        axis_plot.set_axisbelow(True)
        string_sensitivity_title = wrap_label(
            "Record- versus Event-Weighted Prevalence: "
            f"T{integer_sensitivity_topic_id} {string_sensitivity_topic_label}",
            integer_width=75,
        )

    axis_plot.set_title(string_sensitivity_title)
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_07_record_event_sensitivity")

    data_frame_map = pandas.DataFrame({
        "umap_1": array_sentence_embeddings_2d[:, 0],
        "umap_2": array_sentence_embeddings_2d[:, 1],
        "topic": data_frame_document_topics["bertopic_Topic"].to_numpy(),
    })
    list_substantive_topic_ids = sorted(data_frame_map.loc[data_frame_map["topic"] != -1, "topic"].unique())
    figure_plot, axis_plot = plt.subplots(figsize=(11, 9))
    data_frame_complex_points = data_frame_map[data_frame_map["topic"] == -1]
    axis_plot.scatter(
        data_frame_complex_points["umap_1"],
        data_frame_complex_points["umap_2"],
        s=5,
        alpha=0.12,
        color="#777777",
        linewidth=0,
        rasterized=True,
    )

    for integer_topic_position, integer_topic_id in enumerate(list_substantive_topic_ids):
        data_frame_topic_points = data_frame_map[data_frame_map["topic"] == integer_topic_id]
        axis_plot.scatter(
            data_frame_topic_points["umap_1"],
            data_frame_topic_points["umap_2"],
            s=6,
            alpha=0.48,
            color=list_topic_colours[integer_topic_position % len(list_topic_colours)],
            linewidth=0,
            rasterized=True,
        )

    list_top_map_topics = data_frame_map.loc[data_frame_map["topic"] != -1, "topic"].value_counts().head(15).index

    for integer_topic_id in list_top_map_topics:
        data_frame_topic_points = data_frame_map[data_frame_map["topic"] == integer_topic_id]
        axis_plot.annotate(
            f"T{int(integer_topic_id)}",
            (data_frame_topic_points["umap_1"].median(), data_frame_topic_points["umap_2"].median()),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#555555", "linewidth": 0.5, "alpha": 0.85},
        )

    axis_plot.set_title("Two-Dimensional Projection of Unique Recall Narratives")
    axis_plot.set_xlabel("UMAP dimension 1")
    axis_plot.set_ylabel("UMAP dimension 2")
    figure_plot.tight_layout()
    save_figure(figure_plot, "figure_16_08_semantic_topic_map")

    return list_figure_paths


def write_experiment_artifacts(
    string_experiment_root,
    data_frame_full_records,
    data_frame_year_audit,
    data_frame_semantic_source,
    data_frame_record_mapping,
    dictionary_pipeline_outputs,
    list_manufacturer_terms,
    integer_start_year,
    integer_end_year,
    float_outlier_threshold,
    function_progress_callback=None,
):
    string_experiment_root = Path(string_experiment_root).resolve()
    dictionary_experiment_paths = build_experiment_paths(string_experiment_root)

    for string_directory in [
        dictionary_experiment_paths["dashboard_records"].parent,
        dictionary_experiment_paths["document_topics"].parent,
        dictionary_experiment_paths["manual_labels"].parent,
        dictionary_experiment_paths["static_figures"],
    ]:
        string_directory.mkdir(parents=True, exist_ok=True)

    data_frame_dashboard_records = build_pipeline_dashboard_records(
        data_frame_semantic_source=data_frame_semantic_source,
        data_frame_document_topics=dictionary_pipeline_outputs["document_topics"],
        dictionary_automatic_topic_labels=(dictionary_pipeline_outputs["automatic_topic_labels"]),
    )
    list_topic_label_rows = []

    for integer_topic_id, string_topic_label in sorted(dictionary_pipeline_outputs["automatic_topic_labels"].items()):
        list_topic_label_rows.append({
            "topic": int(integer_topic_id),
            "manual_label": string_topic_label,
            "present_in_current_model": True,
            "label_source": "automatic_top_keywords",
        })

    data_frame_topic_labels = pandas.DataFrame(list_topic_label_rows)
    data_frame_cluster_summary = calculate_experiment_cluster_summary(
        data_frame_dashboard_records=data_frame_dashboard_records,
        dictionary_topic_label_lookup=(dictionary_pipeline_outputs["automatic_topic_labels"]),
    )
    data_frame_document_topics = dictionary_pipeline_outputs["document_topics"].copy()
    data_frame_semantic_search_index = (
        data_frame_document_topics[
            [
                "semantic_text_id",
                "semantic_text",
                "occurrence_count",
                "embedding_row",
                "token_length",
                "will_be_truncated",
                "bertopic_Topic",
                "bertopic_Name",
                "bertopic_CustomName",
            ]
        ]
        .rename(
            columns={"bertopic_Topic": "bertopic_topic", "bertopic_Name": "bertopic_name", "bertopic_CustomName": "manual_topic_label"}
        )
    )

    update_experiment_progress(function_progress_callback, 0.82, "Saving validated dashboard artifacts")
    data_frame_dashboard_records.to_parquet(dictionary_experiment_paths["dashboard_records"], index=False)
    data_frame_document_topics.to_parquet(dictionary_experiment_paths["document_topics"], index=False)
    data_frame_record_mapping.to_parquet(dictionary_experiment_paths["record_mapping"], index=False)
    numpy.save(dictionary_experiment_paths["embeddings"], dictionary_pipeline_outputs["sentence_embeddings"])
    numpy.save(
        dictionary_experiment_paths["manual_labels"].parent / "sentence_embeddings_2d.npy",
        dictionary_pipeline_outputs["sentence_embeddings_2d"],
    )
    numpy.save(
        dictionary_experiment_paths["manual_labels"].parent /
        "topics_original_unsupervised.npy",
        dictionary_pipeline_outputs["original_topics"],
    )
    numpy.save(
        dictionary_experiment_paths["manual_labels"].parent /
        "topics_after_outlier_reassignment.npy",
        dictionary_pipeline_outputs["final_topics"],
    )
    data_frame_topic_labels.to_csv(dictionary_experiment_paths["manual_labels"], index=False)
    dictionary_pipeline_outputs["topic_keywords"].to_parquet(dictionary_experiment_paths["topic_keywords"], index=False)
    dictionary_pipeline_outputs["representative_documents"].to_parquet(
        dictionary_experiment_paths["representative_documents"],
        index=False,
    )
    dictionary_pipeline_outputs["topic_information"].to_parquet(
        dictionary_experiment_paths["manual_labels"].parent /
        "bertopic_unsupervised_topic_info.parquet",
        index=False,
    )
    data_frame_semantic_search_index.to_parquet(dictionary_experiment_paths["semantic_search_index"], index=False)
    data_frame_cluster_summary.to_csv(dictionary_experiment_paths["cluster_summary"], index=False)
    data_frame_year_audit.to_csv(dictionary_experiment_paths["manual_labels"].parent / "download_year_audit.csv", index=False)

    dictionary_semantic_search_config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model_name": EXPERIMENT_MODEL_NAME,
        "embedding_dimension": int(dictionary_pipeline_outputs["sentence_embeddings"].shape[1]),
        "normalize_embeddings": True,
        "semantic_text_column": "semantic_text",
        "semantic_text_id_column": "semantic_text_id",
        "masked_words": PIPELINE_GENERIC_MASKED_WORDS,
    }
    dictionary_experiment_paths["semantic_search_config"].write_text(
        json.dumps(dictionary_semantic_search_config, indent=2),
        encoding="utf-8",
    )
    dictionary_experiment_paths["manufacturer_terms"].write_text(json.dumps(list_manufacturer_terms, indent=2), encoding="utf-8")

    dictionary_silhouette_metrics = dictionary_pipeline_outputs["silhouette_metrics"]
    list_package_names = ["bertopic", "umap-learn", "hdbscan", "sentence-transformers", "scikit-learn", "numpy", "pandas"]
    dictionary_package_versions = {}

    for string_package_name in list_package_names:
        dictionary_package_versions[string_package_name] = importlib.metadata.version(string_package_name)

    dictionary_experiment_metadata = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "unsupervised_bertopic_dashboard_pipeline",
        "label_source": "automatic_top_keywords_not_manually_reviewed",
        "year_range": [int(integer_start_year), int(integer_end_year)],
        "downloaded_years": data_frame_year_audit.to_dict(orient="records"),
        "openfda_download": {
            "page_limit": OPENFDA_PAGE_LIMIT,
            "request_timeout_seconds": (OPENFDA_REQUEST_TIMEOUT_SECONDS),
            "request_pause_seconds": OPENFDA_REQUEST_PAUSE_SECONDS,
            "accepted_date_formats": ["YYYY-MM-DD", "YYYYMMDD"],
            "year_maximum_attempts": OPENFDA_YEAR_MAX_ATTEMPTS,
            "year_retry_wait_seconds": (OPENFDA_YEAR_RETRY_WAIT_SECONDS),
        },
        "number_of_full_records": int(len(data_frame_dashboard_records)),
        "number_of_unique_narratives": int(len(data_frame_document_topics)),
        "embedding_model": EXPERIMENT_MODEL_NAME,
        "embedding_shape": list(dictionary_pipeline_outputs["sentence_embeddings"].shape),
        "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
        "hdbscan": {
            "min_cluster_size": 40,
            "min_samples": 10,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.95, "max_features": 30_000},
        "ctfidf": {"reduce_frequent_words": True, "bm25_weighting": False},
        "outlier_reassignment": {
            "strategy": "embeddings",
            "threshold": float(float_outlier_threshold),
            "number_of_outliers_before": int(numpy.sum(dictionary_pipeline_outputs["original_topics"] == -1)),
            "number_of_outliers_after": int(numpy.sum(dictionary_pipeline_outputs["final_topics"] == -1)),
        },
        "number_of_topics": int(len(set(dictionary_pipeline_outputs["final_topics"].tolist()) - {-1})),
        "silhouette_metrics": dictionary_silhouette_metrics,
        "cosine_silhouette": dictionary_silhouette_metrics["original_embedding_cosine"],
        "reduced_space_silhouette": dictionary_silhouette_metrics["original_reduced_euclidean"],
        "final_cosine_silhouette": dictionary_silhouette_metrics["final_embedding_cosine"],
        "final_reduced_space_silhouette": dictionary_silhouette_metrics["final_reduced_euclidean"],
        "package_versions": dictionary_package_versions,
    }
    dictionary_experiment_paths["experiment_metadata"].write_text(json.dumps(dictionary_experiment_metadata, indent=2), encoding="utf-8")

    update_experiment_progress(function_progress_callback, 0.88, "Generating static analytical figures")
    create_experiment_static_figures(
        data_frame_dashboard_records=data_frame_dashboard_records,
        data_frame_document_topics=data_frame_document_topics,
        array_sentence_embeddings_2d=(dictionary_pipeline_outputs["sentence_embeddings_2d"]),
        string_output_directory=dictionary_experiment_paths["static_figures"],
    )
    return dictionary_experiment_paths, dictionary_experiment_metadata, data_frame_cluster_summary


def write_experiment_manifest(string_experiment_root):
    string_experiment_root = Path(string_experiment_root).resolve()
    list_artifact_records = []

    for string_artifact_path in sorted(string_experiment_root.rglob("*")):
        if not string_artifact_path.is_file():
            continue

        string_relative_path = string_artifact_path.relative_to(string_experiment_root).as_posix()

        if string_relative_path == EXPERIMENT_MANIFEST_FILE:
            continue

        list_artifact_records.append({
            "path": string_relative_path,
            "size_bytes": int(string_artifact_path.stat().st_size),
            "sha256": calculate_file_sha256(string_artifact_path),
        })

    dictionary_manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": int(len(list_artifact_records)),
        "artifacts": list_artifact_records,
    }
    string_manifest_path = string_experiment_root / EXPERIMENT_MANIFEST_FILE
    string_manifest_path.write_text(json.dumps(dictionary_manifest, indent=2), encoding="utf-8")

    return dictionary_manifest


def validate_experiment_directory(string_experiment_root):
    string_experiment_root = Path(string_experiment_root).resolve()
    string_manifest_path = string_experiment_root / EXPERIMENT_MANIFEST_FILE

    if not string_manifest_path.exists():
        raise ValueError("The experiment manifest is missing.")

    dictionary_manifest = json.loads(string_manifest_path.read_text(encoding="utf-8"))

    if int(dictionary_manifest.get("schema_version", -1)) != (EXPERIMENT_SCHEMA_VERSION):
        raise ValueError("The experiment schema version is unsupported.")

    dictionary_artifact_lookup = {}

    for dictionary_artifact in dictionary_manifest.get("artifacts", []):
        string_artifact_relative_path = dictionary_artifact["path"]
        dictionary_artifact_lookup[string_artifact_relative_path] = dictionary_artifact

    for string_required_path in EXPERIMENT_REQUIRED_RELATIVE_PATHS:
        if string_required_path not in dictionary_artifact_lookup:
            raise ValueError(f"Required experiment artifact is missing: {string_required_path}")

    for string_relative_path, dictionary_artifact in dictionary_artifact_lookup.items():
        object_relative_path = PurePosixPath(string_relative_path)

        if (object_relative_path.is_absolute() or ".." in object_relative_path.parts):
            raise ValueError("The experiment contains an unsafe path.")

        string_artifact_path = string_experiment_root.joinpath(*object_relative_path.parts)

        if not string_artifact_path.is_file():
            raise ValueError(f"Missing artifact: {string_relative_path}")

        if string_artifact_path.stat().st_size != int(dictionary_artifact["size_bytes"]):
            raise ValueError(f"File-size mismatch: {string_relative_path}")

        if calculate_file_sha256(string_artifact_path) != dictionary_artifact["sha256"]:
            raise ValueError(f"Checksum mismatch: {string_relative_path}")

    dictionary_experiment_paths = build_experiment_paths(string_experiment_root)
    data_frame_dashboard_records = pandas.read_parquet(dictionary_experiment_paths["dashboard_records"])
    data_frame_document_topics = pandas.read_parquet(dictionary_experiment_paths["document_topics"])
    data_frame_semantic_search_index = pandas.read_parquet(dictionary_experiment_paths["semantic_search_index"])
    array_embeddings = numpy.load(dictionary_experiment_paths["embeddings"], mmap_mode="r")
    data_frame_manual_labels = pandas.read_csv(dictionary_experiment_paths["manual_labels"])
    data_frame_cluster_summary = pandas.read_csv(dictionary_experiment_paths["cluster_summary"])
    dictionary_experiment_metadata = json.loads(dictionary_experiment_paths["experiment_metadata"].read_text(encoding="utf-8"))

    set_required_record_columns = {"semantic_text_id", "bertopic_Topic", "topic_label", "reason_for_recall", "event_date_initiated"}

    if not set_required_record_columns.issubset(data_frame_dashboard_records.columns):
        raise ValueError("Dashboard records have an invalid schema.")

    if not {"topic", "manual_label"}.issubset(data_frame_manual_labels.columns):
        raise ValueError("Topic labels have an invalid schema.")

    set_required_cluster_summary_columns = {
        "topic",
        "topic_label",
        "full_record_count",
        "full_record_percentage",
        "unique_narrative_count",
        "unique_narrative_percentage",
    }

    if not set_required_cluster_summary_columns.issubset(data_frame_cluster_summary.columns):
        raise ValueError("The cluster summary has an invalid schema.")

    if int(data_frame_cluster_summary["full_record_count"].sum()) != len(data_frame_dashboard_records):
        raise ValueError("Cluster record counts do not cover the dataset.")

    if int(data_frame_cluster_summary["unique_narrative_count"].sum()) != len(data_frame_document_topics):
        raise ValueError("Cluster narrative counts do not cover the semantic corpus.")

    for string_percentage_column in ["full_record_percentage", "unique_narrative_percentage"]:
        if not numpy.isclose(data_frame_cluster_summary[string_percentage_column].sum(), 100.0):
            raise ValueError("Cluster percentages do not sum to 100%.")

    set_required_silhouette_metrics = {
        "original_embedding_cosine",
        "original_reduced_euclidean",
        "final_embedding_cosine",
        "final_reduced_euclidean",
    }

    if not set_required_silhouette_metrics.issubset(dictionary_experiment_metadata.get("silhouette_metrics", {})):
        raise ValueError("The silhouette metadata is incomplete.")

    if (
        array_embeddings.ndim != 2
        or len(array_embeddings) != len(data_frame_document_topics)
        or len(array_embeddings) != len(data_frame_semantic_search_index)
        or array_embeddings.shape[1] != 768
    ):
        raise ValueError("Embedding rows do not align with topic artifacts.")

    array_expected_rows = numpy.arange(len(data_frame_document_topics))

    if not numpy.array_equal(data_frame_document_topics["embedding_row"].to_numpy(), array_expected_rows):
        raise ValueError("Document embedding rows are not sequential.")

    if not numpy.array_equal(data_frame_semantic_search_index["embedding_row"].to_numpy(), array_expected_rows):
        raise ValueError("Search-index rows are not sequential.")

    return {
        "manifest": dictionary_manifest,
        "full_records": int(len(data_frame_dashboard_records)),
        "unique_narratives": int(len(data_frame_document_topics)),
        "embedding_dimension": int(array_embeddings.shape[1]),
    }


def create_experiment_archive(string_experiment_root):
    string_experiment_root = Path(string_experiment_root).resolve()
    bytes_io_archive_buffer = BytesIO()

    with zipfile.ZipFile(bytes_io_archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as object_archive:
        for string_artifact_path in sorted(string_experiment_root.rglob("*")):
            if string_artifact_path.is_file():
                object_archive.write(string_artifact_path, string_artifact_path.relative_to(string_experiment_root).as_posix())

    bytes_archive = bytes_io_archive_buffer.getvalue()

    if len(bytes_archive) > MAX_EXPERIMENT_ARCHIVE_BYTES:
        raise RuntimeError("The experiment ZIP exceeds the dashboard upload limit.")

    return bytes_archive


def safely_extract_experiment_archive(bytes_archive, string_destination_root):
    if len(bytes_archive) > MAX_EXPERIMENT_ARCHIVE_BYTES:
        raise ValueError("The uploaded archive is too large.")

    string_destination_root = Path(string_destination_root).resolve()
    string_destination_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BytesIO(bytes_archive), mode="r") as object_archive:
        list_archive_members = object_archive.infolist()

        if len(list_archive_members) > MAX_EXPERIMENT_ARCHIVE_MEMBERS:
            raise ValueError("The archive contains too many files.")

        integer_total_uncompressed_bytes = 0

        for object_archive_member in list_archive_members:
            if not object_archive_member.is_dir():
                integer_total_uncompressed_bytes = integer_total_uncompressed_bytes + object_archive_member.file_size

        if integer_total_uncompressed_bytes > (MAX_EXPERIMENT_UNCOMPRESSED_BYTES):
            raise ValueError("The expanded experiment is too large.")

        for object_archive_member in list_archive_members:
            string_relative_path = PurePosixPath(object_archive_member.filename)

            if (string_relative_path.is_absolute() or ".." in string_relative_path.parts or not string_relative_path.parts):
                raise ValueError("The archive contains an unsafe path.")

            integer_unix_mode = (object_archive_member.external_attr >> 16) & 0o170000

            if integer_unix_mode == 0o120000:
                raise ValueError("Symbolic links are not accepted.")

            string_target_path = string_destination_root.joinpath(*string_relative_path.parts).resolve()

            if string_destination_root not in string_target_path.parents:
                raise ValueError("The archive contains an unsafe path.")

            if object_archive_member.is_dir():
                string_target_path.mkdir(parents=True, exist_ok=True)
                continue

            string_target_path.parent.mkdir(parents=True, exist_ok=True)

            with object_archive.open(object_archive_member, "r") as object_source_file:
                with string_target_path.open("wb") as object_target_file:
                    shutil.copyfileobj(object_source_file, object_target_file)


def replace_active_custom_experiment(string_prepared_experiment_root, bytes_archive=None):
    string_previous_workspace_value = st.session_state.get("custom_experiment_workspace")
    string_new_workspace = Path(tempfile.mkdtemp(prefix="fda_recall_custom_experiment_"))
    string_new_experiment_root = string_new_workspace / "experiment"
    shutil.copytree(string_prepared_experiment_root, string_new_experiment_root)
    dictionary_validation_summary = validate_experiment_directory(string_new_experiment_root)

    st.session_state["custom_experiment_workspace"] = str(string_new_workspace)
    st.session_state["custom_experiment_root"] = str(string_new_experiment_root)
    st.session_state["custom_experiment_validation"] = dictionary_validation_summary
    st.session_state["custom_experiment_archive"] = bytes_archive
    st.session_state["activate_custom_experiment"] = True
    if "semantic_search_request" in st.session_state:
        del st.session_state["semantic_search_request"]
    load_dashboard_data.clear()
    load_semantic_search_data.clear()

    if string_previous_workspace_value:
        string_previous_workspace = Path(string_previous_workspace_value)

        if string_previous_workspace.exists() and string_previous_workspace != (string_new_workspace):
            shutil.rmtree(string_previous_workspace, ignore_errors=True)

    return dictionary_validation_summary


def build_full_custom_experiment(
    integer_start_year,
    integer_end_year,
    float_outlier_threshold,
    string_openfda_api_key=None,
    string_huggingface_token=None,
    function_progress_callback=None,
):
    string_build_workspace = Path(tempfile.mkdtemp(prefix="fda_recall_experiment_build_"))
    string_experiment_root = string_build_workspace / "experiment"

    try:
        data_frame_full_records, data_frame_year_audit = download_openfda_year_range(
            integer_start_year=integer_start_year,
            integer_end_year=integer_end_year,
            string_api_key=string_openfda_api_key,
            function_progress_callback=function_progress_callback,
        )
        list_manufacturer_terms = json.loads(
            (
                string_default_project_root /
                "reports" /
                "bertopic_unsupervised_mpnet" /
                "semantic_search_manufacturer_terms.json"
            ).read_text(encoding="utf-8")
        )
        (data_frame_semantic_source, data_frame_unique_semantic_corpus, data_frame_record_mapping) = build_unique_semantic_corpus(
            data_frame_full_records=data_frame_full_records,
            list_manufacturer_terms=list_manufacturer_terms,
            function_progress_callback=function_progress_callback,
        )
        dictionary_pipeline_outputs = fit_fixed_bertopic_pipeline(
            data_frame_unique_semantic_corpus=data_frame_unique_semantic_corpus,
            float_outlier_threshold=float_outlier_threshold,
            string_huggingface_token=string_huggingface_token,
            function_progress_callback=function_progress_callback,
        )
        write_experiment_artifacts(
            string_experiment_root=string_experiment_root,
            data_frame_full_records=data_frame_full_records,
            data_frame_year_audit=data_frame_year_audit,
            data_frame_semantic_source=data_frame_semantic_source,
            data_frame_record_mapping=data_frame_record_mapping,
            dictionary_pipeline_outputs=dictionary_pipeline_outputs,
            list_manufacturer_terms=list_manufacturer_terms,
            integer_start_year=integer_start_year,
            integer_end_year=integer_end_year,
            float_outlier_threshold=float_outlier_threshold,
            function_progress_callback=function_progress_callback,
        )
        write_experiment_manifest(string_experiment_root)
        dictionary_validation_summary = validate_experiment_directory(string_experiment_root)
        update_experiment_progress(function_progress_callback, 0.96, "Compressing the completed experiment")
        bytes_archive = create_experiment_archive(string_experiment_root)
        update_experiment_progress(function_progress_callback, 1.0, "Experiment completed")

        return string_experiment_root, bytes_archive, dictionary_validation_summary
    except Exception:
        shutil.rmtree(string_build_workspace, ignore_errors=True)
        raise


def load_experiment_report_inputs(string_experiment_root):
    dictionary_experiment_paths = build_experiment_paths(string_experiment_root)

    with dictionary_experiment_paths["experiment_metadata"].open("r", encoding="utf-8") as object_metadata_file:
        dictionary_metadata = json.load(object_metadata_file)

    if dictionary_experiment_paths["cluster_summary"].exists():
        data_frame_cluster_summary = pandas.read_csv(dictionary_experiment_paths["cluster_summary"])
    else:
        data_frame_report_records = pandas.read_parquet(dictionary_experiment_paths["dashboard_records"])
        dictionary_report_label_lookup = (
            pandas.read_csv(dictionary_experiment_paths["manual_labels"])
            .set_index("topic")["manual_label"]
            .to_dict()
        )
        data_frame_cluster_summary = calculate_experiment_cluster_summary(
            data_frame_dashboard_records=data_frame_report_records,
            dictionary_topic_label_lookup=dictionary_report_label_lookup,
        )

        dictionary_metadata.setdefault("number_of_full_records", int(len(data_frame_report_records)))
        dictionary_metadata.setdefault("number_of_unique_narratives", int(data_frame_report_records["semantic_text_id"].nunique()))
    list_available_figures = sorted(dictionary_experiment_paths["static_figures"].glob("*.png"))

    return dictionary_metadata, data_frame_cluster_summary, list_available_figures


def build_experiment_pdf_report(string_experiment_name, dictionary_metadata, data_frame_cluster_summary, list_figure_paths):
    string_cache_root = Path(tempfile.gettempdir()) / "fda_recall_ml_cache"
    string_matplotlib_cache = string_cache_root / "matplotlib"
    string_matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(string_matplotlib_cache))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bytes_io_report_buffer = BytesIO()
    dictionary_silhouette_metrics = dictionary_metadata.get("silhouette_metrics", {})

    with PdfPages(bytes_io_report_buffer) as object_pdf_document:
        figure_plot = plt.figure(figsize=(8.27, 11.69))
        axis_plot = figure_plot.add_axes([0.08, 0.08, 0.84, 0.84])
        axis_plot.axis("off")
        axis_plot.text(0, 1, "FDA Medical Device Recall Topic Report", fontsize=20, fontweight="bold", va="top")
        axis_plot.text(0, 0.94, string_experiment_name, fontsize=13, color="#3B6F8F", va="top")
        list_summary_lines = [
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Full FDA records: {dictionary_metadata.get('number_of_full_records', 'Not available')}",
            f"Unique semantic narratives: {dictionary_metadata.get('number_of_unique_narratives', 'Not available')}",
            f"Substantive BERTopic clusters: {dictionary_metadata.get('number_of_topics', 'Not available')}",
            f"Embedding model: {dictionary_metadata.get('embedding_model', 'Not available')}",
        ]
        axis_plot.text(0, 0.86, "\n".join(list_summary_lines), fontsize=11, va="top", linespacing=1.6)
        axis_plot.text(0, 0.66, "Silhouette Metrics", fontsize=14, fontweight="bold", va="top")
        list_metric_lines = []

        for string_metric_name, float_metric_value in (dictionary_silhouette_metrics.items()):
            string_readable_name = string_metric_name.replace("_", " ").title()
            string_readable_value = f"{float(float_metric_value):.4f}" if float_metric_value is not None else "Not defined"
            list_metric_lines.append(f"{string_readable_name}: {string_readable_value}")

        if not list_metric_lines:
            dictionary_fallback_metrics = {
                "Embedding cosine": dictionary_metadata.get("cosine_silhouette"),
                "Reduced-space Euclidean": dictionary_metadata.get("reduced_space_silhouette"),
            }
            for string_metric_name, float_metric_value in (dictionary_fallback_metrics.items()):
                string_readable_metric = (
                    f"{string_metric_name}: "
                    + (f"{float(float_metric_value):.4f}" if float_metric_value is not None else "Not available")
                )
                list_metric_lines.append(string_readable_metric)

        axis_plot.text(0, 0.61, "\n".join(list_metric_lines), fontsize=11, va="top", linespacing=1.6)
        axis_plot.text(0, 0.35, "Interpretation Boundary", fontsize=14, fontweight="bold", va="top")
        axis_plot.text(
            0,
            0.30,
            textwrap.fill(
                "Silhouette metrics describe geometric separation in "
                "the analysed corpus. Topic prevalence describes FDA "
                "recall records and does not estimate market incidence, "
                "clinical risk, or manufacturer safety performance.",
                width=90,
            ),
            fontsize=10,
            va="top",
        )
        object_pdf_document.savefig(figure_plot, bbox_inches="tight")
        plt.close(figure_plot)

        data_frame_display_summary = data_frame_cluster_summary.copy()
        list_wrapped_topic_labels = []
        list_full_record_percentages = []
        list_unique_narrative_percentages = []

        for variable_topic_label in data_frame_display_summary["topic_label"]:
            string_wrapped_topic_label = textwrap.fill(str(variable_topic_label), width=44)
            list_wrapped_topic_labels.append(string_wrapped_topic_label)

        for float_percentage in data_frame_display_summary["full_record_percentage"]:
            list_full_record_percentages.append(f"{float(float_percentage):.2f}%")

        for float_percentage in data_frame_display_summary["unique_narrative_percentage"]:
            list_unique_narrative_percentages.append(f"{float(float_percentage):.2f}%")

        data_frame_display_summary["topic_label"] = list_wrapped_topic_labels
        data_frame_display_summary["full_record_percentage"] = list_full_record_percentages
        data_frame_display_summary["unique_narrative_percentage"] = list_unique_narrative_percentages
        list_display_columns = [
            "topic",
            "topic_label",
            "full_record_count",
            "full_record_percentage",
            "unique_narrative_count",
            "unique_narrative_percentage",
        ]
        integer_rows_per_page = 18

        for integer_start_row in range(0, len(data_frame_display_summary), integer_rows_per_page):
            data_frame_page_table = data_frame_display_summary.iloc[
                integer_start_row:integer_start_row + integer_rows_per_page
            ][list_display_columns]
            figure_plot, axis_plot = plt.subplots(figsize=(11.69, 8.27))
            axis_plot.axis("off")
            axis_plot.set_title("BERTopic Cluster Sizes", fontsize=16, fontweight="bold", pad=18)
            table_cluster_summary = axis_plot.table(
                cellText=data_frame_page_table.values,
                colLabels=["Topic", "Label", "Records", "Records %", "Unique narratives", "Unique %"],
                cellLoc="left",
                colLoc="left",
                loc="center",
                colWidths=[0.06, 0.47, 0.10, 0.10, 0.14, 0.10],
            )
            table_cluster_summary.auto_set_font_size(False)
            table_cluster_summary.set_fontsize(7.5)
            table_cluster_summary.scale(1, 1.7)

            for integer_column_index in range(len(list_display_columns)):
                table_cluster_summary[(0, integer_column_index)].set_facecolor("#D9E7EF")
                table_cluster_summary[0, integer_column_index].set_text_props(weight="bold")

            figure_plot.tight_layout()
            object_pdf_document.savefig(figure_plot, bbox_inches="tight")
            plt.close(figure_plot)

        for string_figure_path in list_figure_paths:
            array_image = plt.imread(string_figure_path)
            figure_plot, axis_plot = plt.subplots(figsize=(11.69, 8.27))
            axis_plot.imshow(array_image)
            axis_plot.axis("off")
            axis_plot.set_title(string_figure_path.stem.replace("_", " ").title(), fontsize=14, pad=12)
            figure_plot.tight_layout()
            object_pdf_document.savefig(figure_plot, bbox_inches="tight")
            plt.close(figure_plot)

    return bytes_io_report_buffer.getvalue()


st.markdown('<div style="background-color:#0f766e;color:white;padding:0.85rem 1rem;border-radius:8px;font-weight:700;font-size:1rem;margin-bottom:1rem;">openFDA Project by Carlos Barquero, University of Bath | Masters Dissertation 2026</div>', unsafe_allow_html=True)

object_title_column, object_source_code_button_column = st.columns([5, 1])

with object_title_column:
    st.title("FDA Medical Device Recall Failure-Pattern Explorer")

with object_source_code_button_column:
    st.write("")
    if st.button("Source code", width="stretch"):
        st.session_state["dashboard_mode_selector"] = "Source Code"

st.write("Explore discovered failure patterns and search for semantically similar historical FDA recall narratives.")

st.caption(
    "Percentages describe the analysed FDA recall corpus. "
    "They do not estimate device failure probability, market "
    "incidence, product risk, or manufacturer failure rate."
)

if integer_number_missing_labels > 0:
    st.warning(
        f"{integer_number_missing_labels:,} records do not have a "
        "reviewed manual topic label."
    )


st.sidebar.header("Explorer")

string_selected_explorer = st.sidebar.radio(
    "Dashboard mode",
    options=["Topic Explorer", "Semantic Recall Search", "Data Generation", "Experiment Upload", "Report Generator", "Source Code"],
    key="dashboard_mode_selector",
)

if string_selected_explorer in ["Topic Explorer", "Semantic Recall Search"]:
    boolean_use_unique_narratives = st.sidebar.checkbox(
        "Use unique narratives only",
        value=False,
        help=(
            "Count each semantic_text_id once within the current "
            "analysis group. Leave disabled to preserve FDA "
            "record-level prevalence."
        ),
    )
else:
    boolean_use_unique_narratives = False

st.sidebar.divider()

st.sidebar.caption(f"Active source: {string_selected_experiment_source}")

if string_selected_explorer == "Topic Explorer":

    st.sidebar.subheader("Topic Filters")

    series_available_years = data_frame_records["year_initiated"].dropna().astype(int)

    if series_available_years.empty:
        st.error("No valid recall initiation years are available.")
        st.stop()

    integer_minimum_year = int(series_available_years.min())
    integer_maximum_year = int(series_available_years.max())

    tuple_selected_year_range = st.sidebar.slider(
        "Recall initiation year",
        min_value=integer_minimum_year,
        max_value=integer_maximum_year,
        value=(integer_minimum_year, integer_maximum_year),
    )

    list_device_class_options = sorted(data_frame_records["device_class_clean"].dropna().unique().tolist())

    list_selected_device_classes = st.sidebar.multiselect(
        "Device class",
        options=list_device_class_options,
        default=[],
        help="Leave empty to include all device classes.",
    )

    list_firm_options = data_frame_records["recalling_firm_clean"].value_counts().index.tolist()

    list_selected_firms = st.sidebar.multiselect(
        "Recalling firm",
        options=list_firm_options,
        default=[],
        help="Leave empty to include all recalling firms.",
    )


    data_frame_context = (
        data_frame_records[data_frame_records["year_initiated"].between(tuple_selected_year_range[0], tuple_selected_year_range[1])]
        .copy()
    )

    if list_selected_device_classes:
        data_frame_context = (data_frame_context[data_frame_context["device_class_clean"].isin(list_selected_device_classes)].copy())

    if list_selected_firms:
        data_frame_context = (data_frame_context[data_frame_context["recalling_firm_clean"].isin(list_selected_firms)].copy())

    data_frame_context_all_records = data_frame_context.copy()

    if boolean_use_unique_narratives:
        data_frame_context = (
            data_frame_context
            .sort_values(["event_date_initiated", "cfres_id"], ascending=[False, True])
            .drop_duplicates(subset="semantic_text_id", keep="first")
            .copy()
        )

    string_analysis_unit_plural = "unique narratives" if boolean_use_unique_narratives else "recall records"

    string_analysis_unit_title = "Unique narratives" if boolean_use_unique_narratives else "Recall records"

    if data_frame_context.empty:
        st.warning(
            "No recall records match the current filters. "
            "Broaden the analysis context to continue."
        )
        st.stop()


    list_topic_frequency_order = data_frame_context["bertopic_Topic"].value_counts().index.astype(int).tolist()

    list_available_topic_ids = []

    for integer_topic_id in list_topic_frequency_order:
        if integer_topic_id != -1:
            list_available_topic_ids.append(integer_topic_id)

    if -1 in list_topic_frequency_order:
        list_available_topic_ids.append(-1)

    integer_selected_topic_id = st.sidebar.selectbox("Failure pattern", options=list_available_topic_ids, format_func=format_topic)

    with st.sidebar:
        st.divider()
        st.write("**Current analysis context**")
        st.write(
            f"Years: {tuple_selected_year_range[0]} to "
            f"{tuple_selected_year_range[1]}"
        )
        st.write("Device classes: " + (", ".join(list_selected_device_classes) if list_selected_device_classes else "All"))
        st.write("Firms: " + (", ".join(list_selected_firms) if list_selected_firms else "All"))
        st.write("Analysis unit:", string_analysis_unit_title)
        st.caption(
            "Every prevalence denominator is recalculated from "
            "this filtered context before a topic is selected."
        )


    string_selected_topic_label = dictionary_topic_label_lookup.get(int(integer_selected_topic_id), "Unlabelled topic")

    data_frame_selected_topic = (data_frame_context[data_frame_context["bertopic_Topic"] == integer_selected_topic_id].copy())

    st.header(string_selected_topic_label)
    st.caption(f"BERTopic topic ID: {integer_selected_topic_id}")

    if integer_selected_topic_id == -1:
        st.info(
            "Topic -1 contains narratives outside the substantive "
            "BERTopic clusters. Treat it as an outlier and model-coverage "
            "category, not as one coherent failure mode."
        )

    integer_number_context_units = len(data_frame_context)

    integer_number_topic_units = len(data_frame_selected_topic)

    float_topic_prevalence_percent = 100 * integer_number_topic_units / integer_number_context_units

    set_selected_topic_semantic_ids = set(data_frame_selected_topic["semantic_text_id"])

    integer_number_topic_events = (
        data_frame_context_all_records[
            data_frame_context_all_records["semantic_text_id"].isin(set_selected_topic_semantic_ids)
        ]["res_event_number"]
        .nunique()
    )

    string_context_metric_label = "Unique narratives in context" if boolean_use_unique_narratives else "Records in context"

    string_topic_metric_label = "Topic narratives" if boolean_use_unique_narratives else "Topic records"

    string_event_metric_label = "Linked recall events" if boolean_use_unique_narratives else "Unique recall events"

    object_metric_column_1, object_metric_column_2, object_metric_column_3, object_metric_column_4 = st.columns(4)

    object_metric_column_1.metric(string_context_metric_label, f"{integer_number_context_units:,}")

    object_metric_column_2.metric(string_topic_metric_label, f"{integer_number_topic_units:,}")

    object_metric_column_3.metric("Topic prevalence", f"{float_topic_prevalence_percent:.2f}%")

    object_metric_column_4.metric(string_event_metric_label, f"{integer_number_topic_events:,}")


    (tab_overview, tab_temporal, tab_metadata, tab_narratives) = st.tabs(
        ["Overview", "Temporal Analysis", "Metadata", "Recall Narratives"]
    )


    with tab_overview:
        st.subheader("Topic Interpretation")

        if integer_selected_topic_id != -1:
            data_frame_selected_topic_keywords = (
                data_frame_topic_keywords[data_frame_topic_keywords["topic"] == integer_selected_topic_id]
                .sort_values("rank")
                .head(10)
                .copy()
            )

            st.write("**Top BERTopic terms**")
            st.write(" | ".join(data_frame_selected_topic_keywords["term"].astype(str).tolist()))

            data_frame_keyword_table = (
                data_frame_selected_topic_keywords[["rank", "term", "score"]]
                .rename(columns={"rank": "Rank", "term": "Term", "score": "c-TF-IDF score"})
            )

            st.dataframe(data_frame_keyword_table, width="stretch", hide_index=True)
        else:
            st.write(
                "Keywords are not presented as one coherent topic "
                "description for the outlier category."
            )

        st.info(
            "The manual label is a reviewed interpretation of the "
            "cluster. Keywords and narratives are shown so users can "
            "inspect the evidence behind that interpretation."
        )


    with tab_temporal:
        st.subheader("Annual Topic Prevalence")

        data_frame_yearly_trend = calculate_yearly_topic_trend(
            data_frame_context=(data_frame_context_all_records),
            integer_topic_id=integer_selected_topic_id,
            boolean_use_unique_narratives=(boolean_use_unique_narratives),
        )

        figure_prevalence = px.line(
            data_frame_yearly_trend,
            x="year_initiated",
            y="topic_share_percent",
            markers=True,
            hover_data={"topic_record_count": True, "all_record_count": True, "topic_share_percent": ":.2f"},
            labels={
                "year_initiated": "Recall initiation year",
                "topic_share_percent": "Topic prevalence (%)",
                "topic_record_count": "Topic records",
                "all_record_count": "All context records",
            },
        )

        figure_prevalence.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))

        st.plotly_chart(figure_prevalence, width="stretch")

        st.caption(
            f"Annual prevalence equals selected-topic {string_analysis_unit_plural} "
            f"divided by all {string_analysis_unit_plural} in the same year "
            "and current context."
        )

        st.subheader("Annual Topic Record Count")

        figure_counts = px.bar(
            data_frame_yearly_trend,
            x="year_initiated",
            y="topic_record_count",
            hover_data={"all_record_count": True},
            labels={
                "year_initiated": "Recall initiation year",
                "topic_record_count": "Topic records",
                "all_record_count": "All context records",
            },
        )

        figure_counts.update_layout(margin=dict(l=20, r=20, t=30, b=20))

        st.plotly_chart(figure_counts, width="stretch")

        st.dataframe(data_frame_yearly_trend, width="stretch", hide_index=True)


    with tab_metadata:
        st.subheader("Device Class")

        data_frame_device_analysis = (
            calculate_topic_prevalence_by_group(
                data_frame_context=(data_frame_context_all_records),
                integer_topic_id=integer_selected_topic_id,
                string_group_column="device_class_clean",
                boolean_use_unique_narratives=(boolean_use_unique_narratives),
            )
            .sort_values("topic_prevalence_percent", ascending=False)
        )

        figure_device_class = px.bar(
            data_frame_device_analysis,
            x="device_class_clean",
            y="topic_prevalence_percent",
            hover_data={"topic_record_count": True, "all_record_count": True, "topic_prevalence_percent": ":.2f"},
            labels={
                "device_class_clean": "Device class",
                "topic_prevalence_percent": "Topic prevalence (%)",
                "topic_record_count": "Topic records",
                "all_record_count": "All class records",
            },
        )

        figure_device_class.update_layout(margin=dict(l=20, r=20, t=30, b=20))

        st.plotly_chart(figure_device_class, width="stretch")

        st.caption(
            "Device-class prevalence is P(topic | device class) "
            f"using {string_analysis_unit_plural} in the current context."
        )

        st.subheader("Recalling Firms")

        if boolean_use_unique_narratives:
            series_firm_support_counts = (data_frame_context_all_records.groupby("recalling_firm_clean")["semantic_text_id"].nunique())
        else:
            series_firm_support_counts = data_frame_context_all_records["recalling_firm_clean"].value_counts()

        integer_largest_firm_context = int(series_firm_support_counts.max())

        integer_maximum_firm_threshold = max(1, min(100, integer_largest_firm_context))

        integer_default_firm_threshold = min(30, integer_maximum_firm_threshold)

        integer_minimum_firm_records = st.number_input(
            "Minimum narratives per firm" if boolean_use_unique_narratives else "Minimum records per firm",
            min_value=1,
            max_value=integer_maximum_firm_threshold,
            value=integer_default_firm_threshold,
            step=1,
            help=(
                "A support threshold prevents very small firms from "
                "dominating a prevalence ranking."
            ),
        )

        data_frame_firm_analysis = (
            calculate_topic_prevalence_by_group(
                data_frame_context=(data_frame_context_all_records),
                integer_topic_id=integer_selected_topic_id,
                string_group_column="recalling_firm_clean",
                boolean_use_unique_narratives=(boolean_use_unique_narratives),
            )
        )

        data_frame_firm_analysis = (
            data_frame_firm_analysis[data_frame_firm_analysis["all_record_count"] >= integer_minimum_firm_records]
            .sort_values(["topic_prevalence_percent", "topic_record_count"], ascending=[False, False])
            .head(20)
            .sort_values("topic_prevalence_percent", ascending=True)
        )

        if data_frame_firm_analysis.empty:
            st.info("No firms meet the selected minimum support threshold.")
        else:
            figure_firms = px.bar(
                data_frame_firm_analysis,
                x="topic_prevalence_percent",
                y="recalling_firm_clean",
                orientation="h",
                hover_data={"topic_record_count": True, "all_record_count": True, "topic_prevalence_percent": ":.2f"},
                labels={
                    "recalling_firm_clean": "Recalling firm",
                    "topic_prevalence_percent": "Topic prevalence (%)",
                    "topic_record_count": "Topic records",
                    "all_record_count": "All firm records",
                },
            )

            figure_firms.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))

            st.plotly_chart(figure_firms, width="stretch")

        st.caption(
            f"Firm prevalence describes {string_analysis_unit_plural} in the "
            "filtered FDA corpus. It is not a manufacturer failure-rate "
            "estimate."
        )


    with tab_narratives:
        if integer_selected_topic_id != -1:
            st.subheader("BERTopic Representative Narratives")

            st.caption(
                "These corpus-level narratives were selected by "
                "BERTopic to represent the semantic core of the topic; "
                "they do not change with the contextual filters."
            )

            data_frame_representative_topic_documents = (
                data_frame_representative_documents[data_frame_representative_documents["topic"] == integer_selected_topic_id]
                .sort_values("representative_rank")
            )

            if data_frame_representative_topic_documents.empty:
                st.info(
                    "No saved representative narratives are available "
                    "for this topic."
                )
            else:
                for _, series_representative_row in data_frame_representative_topic_documents.iterrows():
                    integer_representative_rank = int(series_representative_row["representative_rank"])

                    with st.expander(f"Representative narrative {integer_representative_rank}"):
                        st.write(series_representative_row["representative_document"])

        st.subheader(
            "Unique Recall Narratives in the Current Selection" if boolean_use_unique_narratives else "FDA Recall Records in the Current Selection"
        )

        string_sample_mode = st.selectbox("Record sample", options=["Most recent", "Random sample"])

        integer_maximum_records_to_show = min(100, len(data_frame_selected_topic))

        if integer_maximum_records_to_show == 1:
            integer_number_records_to_show = 1

            st.caption(
                "One recall record is available in the current "
                "topic and filter selection."
            )
        else:
            integer_number_records_to_show = st.slider(
                "Number of recall records to display",
                min_value=1,
                max_value=integer_maximum_records_to_show,
                value=min(20, integer_maximum_records_to_show),
            )

        list_record_columns = [
            "semantic_text_id",
            "occurrence_count",
            "product_res_number",
            "res_event_number",
            "event_date_initiated",
            "recalling_firm",
            "device_name",
            "device_class_clean",
            "reason_for_recall",
            "root_cause_description",
        ]

        list_available_record_columns = []

        for string_column in list_record_columns:
            if string_column in data_frame_selected_topic.columns:
                list_available_record_columns.append(string_column)

        if string_sample_mode == "Random sample":
            data_frame_records_to_display = (
                data_frame_selected_topic[list_available_record_columns]
                .sample(n=integer_number_records_to_show, random_state=42)
            )
        else:
            data_frame_records_to_display = (
                data_frame_selected_topic[list_available_record_columns]
                .sort_values("event_date_initiated", ascending=False)
                .head(integer_number_records_to_show)
            )

        st.dataframe(data_frame_records_to_display, width="stretch", hide_index=True)

        bytes_records_csv = data_frame_records_to_display.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download displayed records as CSV",
            data=bytes_records_csv,
            file_name=(f"topic_{integer_selected_topic_id}_filtered_records.csv"),
            mime="text/csv",
        )
elif string_selected_explorer == "Semantic Recall Search":

    st.header("Semantic Recall Search")

    st.write(
        "Enter a new or hypothetical reason-for-recall narrative "
        "to retrieve semantically similar historical FDA medical-device "
        "recall narratives."
    )

    st.caption(
        "The search returns semantic neighbours, not a predicted topic "
        "or a probability of model confidence."
    )

    (
        data_frame_similarity_index,
        array_semantic_corpus_embeddings,
        data_frame_semantic_record_mapping,
        dictionary_semantic_search_config,
        list_semantic_manufacturer_terms,
    ) = load_semantic_search_data(
        str(string_semantic_search_index_path),
        str(string_embeddings_path),
        str(string_record_mapping_path),
        str(string_semantic_search_config_path),
        str(string_manufacturer_terms_path),
    )

    validate_semantic_search_artifacts(
        data_frame_similarity_index=data_frame_similarity_index,
        array_corpus_embeddings=array_semantic_corpus_embeddings,
        data_frame_record_mapping=data_frame_semantic_record_mapping,
        dictionary_search_config=dictionary_semantic_search_config,
    )

    with st.form("semantic_recall_search_form"):
        string_query_narrative = st.text_area(
            "Reason for recall",
            height=180,
            placeholder=(
                "Example: The device may unexpectedly shut down "
                "because of an internal power-supply failure."
            ),
        )

        string_query_recalling_firm = st.text_input(
            "Recalling firm (optional)",
            help=(
                "Providing the firm allows the query to use the same "
                "conservative manufacturer-masking rule as the corpus."
            ),
        )

        integer_number_of_semantic_results = st.slider("Number of similar narratives", min_value=5, max_value=30, value=10)

        boolean_include_complex_narratives = st.checkbox(
            "Include outlier / complex narratives",
            value=True,
            help=(
                "When disabled, candidates assigned to BERTopic -1 "
                "are removed before ranking."
            ),
        )

        boolean_semantic_search_submitted = st.form_submit_button("Find similar recalls", type="primary")

    if boolean_semantic_search_submitted:
        if not string_query_narrative.strip():
            st.warning("Please enter a recall narrative.")
        else:
            string_prepared_query = prepare_reason_for_recall_query(
                string_reason_for_recall=string_query_narrative,
                string_recalling_firm=string_query_recalling_firm if string_query_recalling_firm.strip() else None,
                list_manufacturer_terms=list_semantic_manufacturer_terms,
                list_masked_words=dictionary_semantic_search_config["masked_words"],
            )

            if not string_prepared_query.strip():
                st.warning(
                    "The query contains no meaningful text after "
                    "preprocessing. Please provide a more descriptive "
                    "recall narrative."
                )
            else:
                st.session_state["semantic_search_request"] = {
                    "original_query": string_query_narrative,
                    "prepared_query": string_prepared_query,
                    "number_of_results": integer_number_of_semantic_results,
                    "include_complex_narratives": (boolean_include_complex_narratives),
                }

    dictionary_semantic_search_request = st.session_state.get("semantic_search_request")

    if dictionary_semantic_search_request is not None:
        boolean_clear_search_results = st.button("Clear search results")

        if boolean_clear_search_results:
            if "semantic_search_request" in st.session_state:
                del st.session_state["semantic_search_request"]
            st.rerun()

        with st.expander("View processed search query"):
            st.write("**Original query**")
            st.write(dictionary_semantic_search_request["original_query"])
            st.write("**Processed query**")
            st.write(dictionary_semantic_search_request["prepared_query"])

        with st.spinner("Loading MPNet and searching historical narratives..."):
            model_semantic_embedding = load_embedding_model(dictionary_semantic_search_config["model_name"])

            data_frame_nearest_semantic_narratives = (
                find_nearest_semantic_narratives(
                    string_prepared_query=(dictionary_semantic_search_request["prepared_query"]),
                    model_embedding=model_semantic_embedding,
                    data_frame_similarity_index=(data_frame_similarity_index),
                    array_corpus_embeddings=(array_semantic_corpus_embeddings),
                    integer_number_of_results=(dictionary_semantic_search_request["number_of_results"]),
                    boolean_include_complex_narratives=(dictionary_semantic_search_request["include_complex_narratives"]),
                )
            )

            data_frame_linked_semantic_records = (
                expand_semantic_matches_to_records(
                    data_frame_nearest_narratives=(data_frame_nearest_semantic_narratives),
                    data_frame_record_mapping=(data_frame_semantic_record_mapping),
                    data_frame_full_recall_records=data_frame_records,
                )
            )

        integer_all_linked_record_count = len(data_frame_linked_semantic_records)

        if boolean_use_unique_narratives:
            data_frame_linked_records_to_display = (
                data_frame_linked_semantic_records
                .drop_duplicates(subset="semantic_text_id", keep="first")
                .copy()
            )
        else:
            data_frame_linked_records_to_display = data_frame_linked_semantic_records.copy()

        object_search_metric_column_1, object_search_metric_column_2, object_search_metric_column_3 = st.columns(3)

        object_search_metric_column_1.metric(
            "Top semantic similarity",
            (f"{data_frame_nearest_semantic_narratives['cosine_similarity'].max():.3f}"),
        )

        object_search_metric_column_2.metric("Unique neighbours", f"{len(data_frame_nearest_semantic_narratives):,}")

        object_search_metric_column_3.metric(
            "Linked narrative examples" if boolean_use_unique_narratives else "Linked FDA records",
            f"{len(data_frame_linked_records_to_display):,}",
        )

        data_frame_neighbour_topic_summary = (
            data_frame_nearest_semantic_narratives
            .groupby(["bertopic_topic", "manual_topic_label"], as_index=False)
            .agg(
                neighbour_count=("semantic_text_id", "size"),
                mean_similarity=("cosine_similarity", "mean"),
                maximum_similarity=("cosine_similarity", "max"),
            )
            .sort_values(["neighbour_count", "mean_similarity"], ascending=[False, False])
        )

        data_frame_neighbour_topic_summary["topic_display"] = (
            "T"
            + data_frame_neighbour_topic_summary["bertopic_topic"].astype(str)
            + " | "
            + data_frame_neighbour_topic_summary["manual_topic_label"]
        )

        st.subheader("Topic Composition of Nearest Historical Neighbours")

        figure_neighbour_topics = px.bar(
            data_frame_neighbour_topic_summary.sort_values("neighbour_count", ascending=True),
            x="neighbour_count",
            y="topic_display",
            orientation="h",
            hover_data={"mean_similarity": ":.3f", "maximum_similarity": ":.3f"},
            labels={
                "neighbour_count": "Nearest narratives",
                "topic_display": "Historical BERTopic topic",
                "mean_similarity": "Mean similarity",
                "maximum_similarity": "Maximum similarity",
            },
        )

        figure_neighbour_topics.update_layout(
            height=max(320, 55 * len(data_frame_neighbour_topic_summary)),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(figure_neighbour_topics, width="stretch")

        st.caption(
            "This chart describes topic composition among the nearest "
            "historical neighbours. It is not a predicted-topic "
            "probability distribution."
        )

        st.subheader("Most Similar Historical Narratives")

        for _, series_semantic_row in (data_frame_nearest_semantic_narratives.iterrows()):
            integer_similarity_rank = int(series_semantic_row["similarity_rank"])
            float_similarity_value = float(series_semantic_row["cosine_similarity"])
            integer_semantic_topic_id = int(series_semantic_row["bertopic_topic"])
            string_semantic_topic_label = series_semantic_row["manual_topic_label"]

            with st.expander(
                (
                    f"#{integer_similarity_rank} | Similarity "
                    f"{float_similarity_value:.3f} | "
                    f"{string_semantic_topic_label}"
                )
            ):
                st.write("**Historical narrative**")
                st.write(series_semantic_row["semantic_text"])
                st.write("**BERTopic topic ID:**", integer_semantic_topic_id)
                st.write("**Historical record occurrences:**", int(series_semantic_row["occurrence_count"]))

        st.subheader("Linked Historical FDA Recall Records")

        list_semantic_display_columns = [
            "similarity_rank",
            "cosine_similarity",
            "manual_topic_label",
            "product_res_number",
            "res_event_number",
            "event_date_initiated",
            "recalling_firm",
            "device_name",
            "medical_specialty_description",
            "device_class_clean",
            "product_code",
            "root_cause_description",
            "reason_for_recall",
        ]

        list_available_semantic_display_columns = []

        for string_column in list_semantic_display_columns:
            if string_column in data_frame_linked_records_to_display.columns:
                list_available_semantic_display_columns.append(string_column)

        st.dataframe(data_frame_linked_records_to_display[list_available_semantic_display_columns], width="stretch", hide_index=True)

        if boolean_use_unique_narratives:
            st.caption(
                f"{len(data_frame_nearest_semantic_narratives):,} unique semantic "
                f"neighbours map to {integer_all_linked_record_count:,} FDA "
                "records. The unique-narrative filter shows one recent "
                "record example for each matched narrative."
            )
        else:
            st.caption(
                f"{len(data_frame_nearest_semantic_narratives):,} unique semantic "
                f"neighbours map to {integer_all_linked_record_count:,} FDA "
                "records because repeated historical narratives may "
                "occur in multiple records."
            )

        bytes_semantic_records_csv = (
            data_frame_linked_records_to_display[list_available_semantic_display_columns]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download linked records as CSV",
            data=bytes_semantic_records_csv,
            file_name="semantic_recall_search_results.csv",
            mime="text/csv",
        )
    else:
        st.info(
            f"Submit a narrative to search "
            f"{len(data_frame_similarity_index):,} unique historical "
            "semantic narratives. The embedding model loads only "
            "when a search is requested."
        )

elif string_selected_explorer == "Data Generation":
    st.header("Data Generation")
    st.write(
        "Run the dissertation's fixed openFDA, MPNet, UMAP, HDBSCAN, "
        "BERTopic, and outlier-reassignment pipeline without editing "
        "notebook code."
    )
    st.caption(
        "This is a local, compute-intensive workflow. Keep this browser "
        "session open until the downloadable experiment ZIP appears."
    )

    string_generation_message = st.session_state.get("custom_experiment_message", None)

    if string_generation_message:
        st.success(string_generation_message)
        st.session_state["custom_experiment_message"] = None

    integer_current_year = datetime.now().year

    with st.form("custom_experiment_generation_form"):
        object_year_column_1, object_year_column_2 = st.columns(2)

        with object_year_column_1:
            integer_generation_start_year = st.number_input(
                "Start year",
                min_value=2002,
                max_value=integer_current_year,
                value=max(2018, integer_current_year - 7),
                step=1,
            )

        with object_year_column_2:
            integer_generation_end_year = st.number_input(
                "End year",
                min_value=2002,
                max_value=integer_current_year,
                value=min(2025, integer_current_year),
                step=1,
            )

        boolean_use_openfda_api_key = st.checkbox("Use FDA API key", value=False)
        string_openfda_api_key_input = st.text_input(
            "FDA API key",
            type="password",
            help=(
                "This field remains editable. The key is attempted only "
                "when the checkbox is selected; an unaccepted key is "
                "ignored so the download can continue without it."
            ),
        )
        boolean_use_huggingface_token = st.checkbox("Use Hugging Face access token", value=False)
        string_huggingface_token_input = st.text_input(
            "Hugging Face access token",
            type="password",
            help=(
                "This field remains editable. The token is attempted only "
                "when the checkbox is selected; an unaccepted token is "
                "ignored so the public model can load without it."
            ),
        )
        float_generation_outlier_threshold = st.slider(
            "Outlier reassignment threshold",
            min_value=0.55,
            max_value=0.75,
            value=0.63,
            step=0.01,
            help=(
                "Only this BERTopic setting is variable. MPNet, UMAP, "
                "HDBSCAN, vectorizer, and c-TF-IDF settings are fixed."
            ),
        )
        boolean_start_and_save = st.form_submit_button("Start and Save", type="primary")

    if boolean_start_and_save:
        boolean_generation_inputs_are_valid = True
        string_generated_experiment_root = None

        if int(integer_generation_start_year) > int(integer_generation_end_year):
            boolean_generation_inputs_are_valid = False

        if not boolean_generation_inputs_are_valid:
            st.error("An error occurred. Please try again.")
        else:
            string_openfda_api_key_for_run = None
            if boolean_use_openfda_api_key:
                string_openfda_api_key_for_run = string_openfda_api_key_input.strip() or None

            string_huggingface_token_for_run = None
            if boolean_use_huggingface_token:
                string_huggingface_token_for_run = string_huggingface_token_input.strip() or None

            object_generation_progress = st.progress(0.0)
            object_generation_status = st.empty()

            def show_generation_progress(float_fraction, string_stage):
                object_generation_progress.progress(float(float_fraction))
                object_generation_status.write(string_stage)

            try:
                (
                    string_generated_experiment_root,
                    bytes_generated_archive,
                    dictionary_generated_validation,
                ) = build_full_custom_experiment(
                    integer_start_year=int(integer_generation_start_year),
                    integer_end_year=int(integer_generation_end_year),
                    float_outlier_threshold=float(float_generation_outlier_threshold),
                    string_openfda_api_key=string_openfda_api_key_for_run,
                    string_huggingface_token=string_huggingface_token_for_run,
                    function_progress_callback=show_generation_progress,
                )
                replace_active_custom_experiment(
                    string_prepared_experiment_root=(string_generated_experiment_root),
                    bytes_archive=bytes_generated_archive,
                )
                shutil.rmtree(string_generated_experiment_root.parent, ignore_errors=True)
                st.session_state["generated_experiment_archive"] = bytes_generated_archive
                st.session_state["generated_experiment_file_name"] = (
                    f"fda_recall_experiment_"
                    f"{int(integer_generation_start_year)}_"
                    f"{int(integer_generation_end_year)}.zip"
                )
                st.session_state["generated_experiment_validation"] = dictionary_generated_validation
                st.session_state["custom_experiment_message"] = (
                    "The custom experiment is active and ready to "
                    "download."
                )
                object_generation_status.write("Experiment completed")
                st.rerun()
            except Exception:
                object_app_logger.exception("Custom experiment generation failed.")

                if string_generated_experiment_root is not None:
                    shutil.rmtree(Path(string_generated_experiment_root).parent, ignore_errors=True)

                object_generation_progress.empty()
                object_generation_status.empty()
                st.error("An error occurred. Please try again.")

    bytes_generated_archive_from_session = st.session_state.get("generated_experiment_archive")

    if bytes_generated_archive_from_session:
        dictionary_generated_validation = st.session_state.get("generated_experiment_validation", {})
        object_metric_column_1, object_metric_column_2 = st.columns(2)
        object_metric_column_1.metric("FDA records", f"{dictionary_generated_validation.get('full_records', 0):,}")
        object_metric_column_2.metric("Unique narratives", f"{dictionary_generated_validation.get('unique_narratives', 0):,}")
        st.download_button(
            "Download experiment ZIP",
            data=bytes_generated_archive_from_session,
            file_name=st.session_state.get("generated_experiment_file_name", "fda_recall_experiment.zip"),
            mime="application/zip",
            type="primary",
        )

elif string_selected_explorer == "Experiment Upload":
    st.header("Experiment Upload")
    st.write(
        "Upload a ZIP produced by the Data Generation screen. The "
        "archive is validated before it can replace the active custom "
        "experiment."
    )
    st.caption(
        "Only one custom experiment is retained in this dashboard "
        "session. Uploading another valid ZIP replaces it."
    )

    string_upload_message = st.session_state.get("custom_experiment_message", None)

    if string_upload_message:
        st.success(string_upload_message)
        st.session_state["custom_experiment_message"] = None

    object_uploaded_experiment_file = st.file_uploader("Custom experiment ZIP", type=["zip"], accept_multiple_files=False)

    if st.button("Validate and Activate", type="primary", disabled=object_uploaded_experiment_file is None):
        string_upload_workspace = None

        try:
            bytes_uploaded_experiment = object_uploaded_experiment_file.getvalue()
            string_upload_workspace = Path(tempfile.mkdtemp(prefix="fda_recall_uploaded_experiment_"))
            string_extracted_root = string_upload_workspace / "experiment"
            safely_extract_experiment_archive(bytes_archive=bytes_uploaded_experiment, string_destination_root=string_extracted_root)
            dictionary_validation_summary = validate_experiment_directory(string_extracted_root)
            replace_active_custom_experiment(
                string_prepared_experiment_root=string_extracted_root,
                bytes_archive=bytes_uploaded_experiment,
            )
            st.session_state["generated_experiment_archive"] = bytes_uploaded_experiment
            st.session_state["generated_experiment_file_name"] = object_uploaded_experiment_file.name
            st.session_state["generated_experiment_validation"] = dictionary_validation_summary
            st.session_state["custom_experiment_message"] = "The custom experiment was validated and activated."
            st.rerun()
        except Exception:
            object_app_logger.exception("Experiment upload validation failed.")
            st.error("An error occurred. Please try again.")
        finally:
            if string_upload_workspace is not None:
                shutil.rmtree(string_upload_workspace, ignore_errors=True)

    if boolean_custom_experiment_is_available:
        dictionary_custom_validation = st.session_state.get("custom_experiment_validation", {})
        st.subheader("Current Custom Experiment")
        object_metric_column_1, object_metric_column_2 = st.columns(2)
        object_metric_column_1.metric("FDA records", f"{dictionary_custom_validation.get('full_records', 0):,}")
        object_metric_column_2.metric("Unique narratives", f"{dictionary_custom_validation.get('unique_narratives', 0):,}")

elif string_selected_explorer == "Report Generator":
    st.header("Report Generator")
    st.write(
        "Create a PDF containing silhouette metrics, absolute and "
        "relative BERTopic cluster sizes, and the active experiment's "
        "static figures."
    )

    try:
        (
            dictionary_active_report_metadata,
            data_frame_active_cluster_summary,
            list_active_report_figures,
        ) = load_experiment_report_inputs(string_active_project_root)

        dictionary_silhouette_metrics = dictionary_active_report_metadata.get(
            "silhouette_metrics",
            {
                "original_embedding_cosine": (dictionary_active_report_metadata.get("cosine_silhouette")),
                "original_reduced_euclidean": (dictionary_active_report_metadata.get("reduced_space_silhouette")),
            },
        )
        st.subheader("Silhouette Metrics")
        list_metric_columns = st.columns(max(1, min(4, len(dictionary_silhouette_metrics))))

        for integer_metric_position, (string_metric_name, float_metric_value) in enumerate(dictionary_silhouette_metrics.items()):
            list_metric_columns[integer_metric_position % len(list_metric_columns)].metric(
                string_metric_name.replace("_", " ").title(),
                f"{float(float_metric_value):.4f}" if float_metric_value is not None else "Not defined",
            )

        st.subheader("BERTopic Cluster Sizes")
        st.dataframe(data_frame_active_cluster_summary, width="stretch", hide_index=True)

        st.subheader("Static Figures")

        for string_report_figure_path in list_active_report_figures:
            st.image(str(string_report_figure_path), caption=string_report_figure_path.stem.replace("_", " ").title(), width="stretch")

        if not list_active_report_figures:
            st.info(
                "No saved static figures are available for this "
                "experiment. The PDF will contain metrics and tables."
            )

        if st.button("Generate PDF Report", type="primary"):
            try:
                bytes_report_pdf = build_experiment_pdf_report(
                    string_experiment_name=string_selected_experiment_source,
                    dictionary_metadata=dictionary_active_report_metadata,
                    data_frame_cluster_summary=data_frame_active_cluster_summary,
                    list_figure_paths=list_active_report_figures,
                )
                st.session_state["experiment_report_pdf"] = bytes_report_pdf
            except Exception:
                object_app_logger.exception("PDF report generation failed.")
                st.error("An error occurred. Please try again.")

        bytes_report_pdf = st.session_state.get("experiment_report_pdf")

        if bytes_report_pdf:
            st.download_button(
                "Download PDF Report",
                data=bytes_report_pdf,
                file_name="fda_recall_topic_report.pdf",
                mime="application/pdf",
                type="primary",
            )
    except Exception:
        object_app_logger.exception("Report inputs could not be loaded.")
        st.error("An error occurred. Please try again.")

elif string_selected_explorer == "Source Code":
    st.header("Source Code")
    st.write("This screen displays the Python source code for the Streamlit dashboard currently running this app.")
    st.caption(f"Source file: {string_dashboard_source_code_path}")

    try:
        string_dashboard_source_code = string_dashboard_source_code_path.read_text(encoding="utf-8")
        st.download_button("Download source code", data=string_dashboard_source_code.encode("utf-8"), file_name="streamlit_app.py", mime="text/x-python")
        st.code(string_dashboard_source_code, language="python")
    except Exception:
        object_app_logger.exception("Dashboard source code could not be loaded.")
        st.error("An error occurred. Please try again.")


st.divider()
st.caption(
    "Topic and search views are read-only for the active experiment. "
    "Data Generation creates a separate custom experiment and never "
    "changes the dissertation's default artifacts."
)
