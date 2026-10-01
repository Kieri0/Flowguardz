"""Streamlit interface for the same two models used by the FlowGuard site."""

import math
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import PredictionEngine  # noqa: E402


GROUPS = (
    ("connection", "Connection"),
    ("volume", "Packet counts and bytes"),
    ("length", "Packet sizes"),
    ("rate", "Rates and direction"),
    ("timing", "Inter-arrival times"),
    ("tcp", "Protocol, flags and headers"),
    ("activity", "Active and idle periods"),
)


@st.cache_resource
def load_engine():
    with (ROOT / "model.pkl").open("rb") as file:
        return PredictionEngine(pickle.load(file))


def csv_values(file, features):
    try:
        first = pd.read_csv(file, nrows=1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ValueError("The CSV could not be read. Check its header and first data row.") from error
    if first.empty:
        raise ValueError("The CSV needs at least one data row.")
    first.columns = first.columns.str.strip()
    missing = [name for name in features if name not in first.columns]
    if missing:
        raise ValueError(f"The CSV is missing {len(missing)} required columns, including {missing[0]}.")
    values = {}
    for name in features:
        try:
            number = float(first.iloc[0][name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} needs a finite numeric value in the first row.") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} needs a finite numeric value in the first row.")
        values[name] = number
    return values


def group_for(name):
    if name.startswith(("Active ", "Idle ")):
        return "activity"
    if "IAT" in name:
        return "timing"
    if re.search(r"Flag|Header Length|Init_Win|min_seg_size|Protocol", name):
        return "tcp"
    if re.search(r"Bytes/s|Packets/s|Down/Up Ratio", name):
        return "rate"
    if re.search(r"Packet Length|Average Packet Size|Avg .* Segment Size", name):
        return "length"
    if re.search(r"Total .* Packets|Total Length|Subflow|act_data_pkt_fwd", name):
        return "volume"
    return "connection"


def manual_form(features):
    grouped = {key: [] for key, _ in GROUPS}
    for index, name in enumerate(features):
        grouped[group_for(name)].append((index, name))
    values = {}
    with st.form("manual_flow"):
        for group_index, (key, title) in enumerate(GROUPS):
            entries = grouped[key]
            if not entries:
                continue
            with st.expander(f"{title} · {len(entries)} fields", expanded=group_index == 0):
                columns = st.columns(2)
                for position, (index, name) in enumerate(entries):
                    values[name] = columns[position % 2].number_input(
                        name, value=None, step=0.01, key=f"manual_{index}"
                    )
        submitted = st.form_submit_button("Predict flow", type="primary")
    if not submitted:
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Enter all {len(features)} values. First missing: {missing[0]}.")
    return values


def show_results(engine, prediction):
    st.subheader("Model predictions")
    if prediction is None:
        st.info("Upload a CSV or enter the 69 measurements, then select Predict flow.")
        return
    for column, key, title in zip(
        st.columns(2),
        ("random_forest", "logistic_regression"),
        ("Random Forest", "Logistic Regression"),
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.metric("Prediction", prediction[key])
                score = engine.evaluation.get(key, {}).get("accuracy")
                if score is not None:
                    st.caption(f"Held-out accuracy: {score:.4%}")
    if prediction["random_forest"] != prediction["logistic_regression"]:
        st.warning("The models disagree; no combined verdict is assigned.")
    st.caption("This predicts one flow from the study's classes; it does not prove traffic is safe or an attack.")


def main():
    st.set_page_config(page_title="FlowGuard", page_icon="🌸", layout="wide")
    st.title("FlowGuard")
    st.write("Compare Random Forest and Logistic Regression on one network flow using the saved 69-feature study models.")
    try:
        engine = load_engine()
    except (OSError, ValueError, pickle.UnpicklingError) as error:
        st.error(f"Could not load model.pkl: {error}")
        st.stop()

    prediction = None
    st.subheader("Try a sample flow")
    st.caption("Pick a saved example to see its measurements and both model predictions.")
    sample_columns = st.columns(2)
    benign_clicked = sample_columns[0].button("Try BENIGN sample")
    ddos_clicked = sample_columns[1].button("Try DDoS sample")
    sample_name = "BENIGN" if benign_clicked else "DDoS" if ddos_clicked else None
    if sample_name:
        with (ROOT / "examples" / f"{sample_name}.csv").open(encoding="utf-8") as file:
            sample_values = csv_values(file, engine.features)
        st.session_state["input_mode"] = "Enter values manually"
        for index, name in enumerate(engine.features):
            st.session_state[f"manual_{index}"] = float(sample_values[name])
        prediction = engine.predict(sample_values)
        with st.expander(f"View {sample_name} sample measurements"):
            st.dataframe(pd.DataFrame({"Feature": list(sample_values), "Value": list(sample_values.values())}), hide_index=True)

    mode = st.radio("How will you enter the flow?", ("Upload CSV", "Enter values manually"), horizontal=True, key="input_mode")
    input_column, result_column = st.columns([3, 2], gap="large")
    with input_column:
        st.subheader("Flow details")
        st.caption("Use CICIDS2017 column names. Destination Port and identifiers are excluded from this model.")
        if mode == "Upload CSV":
            uploaded = st.file_uploader("Choose a network-flow CSV", type="csv")
            st.caption("The first data row is used. Extra columns are ignored.")
            if st.button("Predict flow", type="primary"):
                if uploaded is None:
                    st.error("Choose a CSV before predicting.")
                else:
                    try:
                        prediction = engine.predict(csv_values(uploaded, engine.features))
                    except ValueError as error:
                        st.error(str(error))
        else:
            try:
                values = manual_form(engine.features)
                if values is not None:
                    prediction = engine.predict(values)
            except ValueError as error:
                st.error(str(error))
    with result_column:
        show_results(engine, prediction)


if __name__ == "__main__":
    main()
