from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("scaling", StandardScaler()),
        ]
    )


def categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("encoding", OneHotEncoder()),
        ]
    )


def clip_outliers(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clipped = dataframe.copy()
    for column in columns:
        percentile_25 = clipped[column].quantile(0.25)
        percentile_75 = clipped[column].quantile(0.75)
        iqr = percentile_75 - percentile_25
        lower_limit = percentile_25 - 1.5 * iqr
        upper_limit = percentile_75 + 1.5 * iqr
        clipped[column] = clipped[column].clip(lower=lower_limit, upper=upper_limit)
    return clipped


def build_choice_mapping(values: pd.Series) -> dict[str, str]:
    unique_values = sorted(
        [value for value in values.dropna().unique().tolist()],
        key=lambda item: str(item).strip().lower(),
    )

    mapping: dict[str, str] = {}
    for raw_value in unique_values:
        label = str(raw_value).strip() or str(raw_value)
        if label in mapping and mapping[label] != raw_value:
            label = str(raw_value)
        mapping[label] = raw_value
    return mapping


def get_default_label(mapping: dict[str, str], raw_value: str) -> str:
    for label, stored_value in mapping.items():
        if stored_value == raw_value:
            return label
    fallback_label = str(raw_value).strip()
    return fallback_label if fallback_label in mapping else next(iter(mapping))


def estimate_water_feature(
    soil_moisture: float,
    rainfall: float,
    temperature: float,
    soil_type: str,
    growth_stage: str,
) -> int:
    irrigation_rule = (
        soil_moisture < 35 and rainfall < 5
    ) or (temperature > 35 and soil_moisture < 40)

    if not irrigation_rule:
        return 0

    water_needed = 20
    if soil_type == "Sandy":
        water_needed += 5
    if temperature > 35:
        water_needed += 5
    if growth_stage == "Mid":
        water_needed += 5
    return water_needed


def estimate_irrigation_amount(
    soil_type: str,
    temperature: float,
    growth_stage: str,
) -> int:
    water_needed = 20
    if soil_type == "Sandy":
        water_needed += 5
    if temperature > 35:
        water_needed += 5
    if growth_stage == "Mid":
        water_needed += 5
    return water_needed


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, object]:
    return {
        "crop": joblib.load(MODEL_DIR / "crop_recommendation_model.pkl"),
        "irrigation": joblib.load(MODEL_DIR / "irrigation_model.pkl"),
        "yield": joblib.load(MODEL_DIR / "yield_prediction_model.pkl"),
    }


@st.cache_resource(show_spinner=False)
def prepare_crop_artifacts() -> dict[str, object]:
    dataframe = pd.read_csv(DATA_DIR / "Crop_Recommendation.csv")
    feature_columns = [
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Temperature",
        "Humidity",
        "pH_Value",
        "Rainfall",
    ]

    encoder = LabelEncoder()
    encoder.fit(dataframe["Crop"])

    stats = {
        column: {
            "min": float(dataframe[column].min()),
            "max": float(dataframe[column].max()),
            "default": float(dataframe[column].median()),
        }
        for column in feature_columns
    }

    return {"encoder": encoder, "feature_columns": feature_columns, "stats": stats}


@st.cache_resource(show_spinner=False)
def prepare_irrigation_artifacts() -> dict[str, object]:
    dataframe = pd.read_csv(DATA_DIR / "irrigation_recommendation_dataset.csv")

    X = dataframe.drop(columns="irrigation_required")
    y = dataframe["irrigation_required"]

    # Use 70% of data for preprocessing setup
    sample_size = int(len(X) * 0.7)
    X_train = X.head(sample_size)

    numeric_columns = [
        "Temperature",
        "Humidity",
        "Rainfall",
        "soil_moisture",
        "water_required_mm",
    ]
    categorical_columns = ["Crop", "soil_type", "growth_stage"]

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline(), numeric_columns),
            ("categorical", categorical_pipeline(), categorical_columns),
        ]
    )
    preprocessor.fit(X_train)

    stats = {
        column: {
            "min": float(dataframe[column].min()),
            "max": float(dataframe[column].max()),
            "default": float(dataframe[column].median()),
        }
        for column in ["Temperature", "Humidity", "Rainfall", "soil_moisture"]
    }

    choices = {
        column: build_choice_mapping(dataframe[column])
        for column in categorical_columns
    }
    defaults = {
        column: dataframe[column].mode().iat[0]
        for column in categorical_columns
    }

    return {
        "preprocessor": preprocessor,
        "stats": stats,
        "choices": choices,
        "defaults": defaults,
    }


@st.cache_resource(show_spinner=False)
def prepare_yield_artifacts() -> dict[str, object]:
    dataframe = pd.read_csv(DATA_DIR / "indian crop production.csv")
    dataframe = dataframe.drop(columns=["District "]).head(20000).copy()
    dataframe = dataframe.dropna(subset=["Production"])
    dataframe = dataframe.dropna(subset=["Crop"])
    dataframe = dataframe.drop(columns=["Production"])
    dataframe = clip_outliers(dataframe, ["Crop_Year", "Area ", "Yield"])
    dataframe = dataframe.reset_index()
    dataframe["Area"] = dataframe["Area "]
    dataframe = dataframe.drop(columns=["Area "])

    X = dataframe.drop(columns=["Yield"])
    y = dataframe["Yield"]

    # Use 80% of data for preprocessing setup
    sample_size = int(len(X) * 0.8)
    X_train = X.head(sample_size)

    numeric_columns = ["Crop_Year", "Area"]
    categorical_columns = ["Crop", "State", "Season"]

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline(), numeric_columns),
            ("categorical", categorical_pipeline(), categorical_columns),
        ]
    )
    preprocessor.fit(X_train)

    stats = {
        "Crop_Year": {
            "min": int(dataframe["Crop_Year"].min()),
            "max": int(dataframe["Crop_Year"].max()),
            "default": int(dataframe["Crop_Year"].median()),
        },
        "Area": {
            "min": float(dataframe["Area"].min()),
            "max": float(dataframe["Area"].max()),
            "default": float(dataframe["Area"].median()),
        },
    }

    choices = {
        column: build_choice_mapping(dataframe[column])
        for column in categorical_columns
    }
    defaults = {
        column: dataframe[column].mode().iat[0]
        for column in categorical_columns
    }

    return {
        "preprocessor": preprocessor,
        "stats": stats,
        "choices": choices,
        "defaults": defaults,
    }


def set_app_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(205, 225, 199, 0.7), transparent 28%),
                    linear-gradient(180deg, #f5f0e3 0%, #edf3ea 100%);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #183a2c 0%, #214a35 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f4f2ea;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.25rem 1.5rem;
                border-radius: 20px;
                background: linear-gradient(135deg, rgba(24, 58, 44, 0.10), rgba(189, 155, 85, 0.18));
                border: 1px solid rgba(24, 58, 44, 0.10);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                color: #183a2c;
            }
            .hero p {
                margin: 0.4rem 0 0;
                color: #40594b;
                font-size: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Smart Agriculture Control Center</h1>
            <p>
                Crop recommendation, irrigation planning, and crop yield forecasting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Models loaded", "3")
    metric_columns[1].metric("Prediction tasks", "Crop, irrigation, yield")
    metric_columns[2].metric("Data source", "Local project files")


def render_crop_page(model: object, artifacts: dict[str, object]) -> None:
    st.subheader("Crop Recommendation")
    st.write("Enter the soil nutrients and weather conditions from the crop dataset.")

    stats = artifacts["stats"]

    with st.form("crop_recommendation_form"):
        col1, col2, col3 = st.columns(3)

        nitrogen = col1.number_input(
            "Nitrogen",
            min_value=int(stats["Nitrogen"]["min"]),
            max_value=int(stats["Nitrogen"]["max"]),
            value=int(round(stats["Nitrogen"]["default"])),
            step=1,
        )
        phosphorus = col2.number_input(
            "Phosphorus",
            min_value=int(stats["Phosphorus"]["min"]),
            max_value=int(stats["Phosphorus"]["max"]),
            value=int(round(stats["Phosphorus"]["default"])),
            step=1,
        )
        potassium = col3.number_input(
            "Potassium",
            min_value=int(stats["Potassium"]["min"]),
            max_value=int(stats["Potassium"]["max"]),
            value=int(round(stats["Potassium"]["default"])),
            step=1,
        )

        col4, col5, col6 = st.columns(3)
        temperature = col4.number_input(
            "Temperature",
            min_value=float(stats["Temperature"]["min"]),
            max_value=float(stats["Temperature"]["max"]),
            value=float(round(stats["Temperature"]["default"], 2)),
            step=0.1,
        )
        humidity = col5.number_input(
            "Humidity",
            min_value=float(stats["Humidity"]["min"]),
            max_value=float(stats["Humidity"]["max"]),
            value=float(round(stats["Humidity"]["default"], 2)),
            step=0.1,
        )
        ph_value = col6.number_input(
            "pH value",
            min_value=float(stats["pH_Value"]["min"]),
            max_value=float(stats["pH_Value"]["max"]),
            value=float(round(stats["pH_Value"]["default"], 2)),
            step=0.1,
        )

        rainfall = st.number_input(
            "Rainfall",
            min_value=float(stats["Rainfall"]["min"]),
            max_value=float(stats["Rainfall"]["max"]),
            value=float(round(stats["Rainfall"]["default"], 2)),
            step=0.1,
        )

        submitted = st.form_submit_button("Recommend crop", use_container_width=True)

    if not submitted:
        return

    input_frame = pd.DataFrame(
        [
            {
                "Nitrogen": nitrogen,
                "Phosphorus": phosphorus,
                "Potassium": potassium,
                "Temperature": temperature,
                "Humidity": humidity,
                "pH_Value": ph_value,
                "Rainfall": rainfall,
            }
        ]
    )

    encoded_prediction = int(model.predict(input_frame)[0])
    predicted_crop = artifacts["encoder"].inverse_transform([encoded_prediction])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(input_frame)[0].max())

    st.success(f"Recommended crop: {predicted_crop}")
    if confidence is not None:
        st.caption(f"Model confidence: {confidence:.1%}")

    st.dataframe(input_frame, use_container_width=True, hide_index=True)


def render_irrigation_page(model: object, artifacts: dict[str, object]) -> None:
    st.subheader("Irrigation Recommendation")
    st.write("Use current field conditions to predict whether irrigation is needed.")

    stats = artifacts["stats"]
    choices = artifacts["choices"]
    defaults = artifacts["defaults"]

    crop_labels = list(choices["Crop"].keys())
    soil_type_labels = list(choices["soil_type"].keys())
    growth_stage_labels = list(choices["growth_stage"].keys())

    with st.form("irrigation_recommendation_form"):
        col1, col2 = st.columns(2)

        crop_label = col1.selectbox(
            "Crop",
            crop_labels,
            index=crop_labels.index(get_default_label(choices["Crop"], defaults["Crop"])),
        )
        soil_type_label = col2.selectbox(
            "Soil type",
            soil_type_labels,
            index=soil_type_labels.index(
                get_default_label(choices["soil_type"], defaults["soil_type"])
            ),
        )

        col3, col4 = st.columns(2)
        growth_stage_label = col3.selectbox(
            "Growth stage",
            growth_stage_labels,
            index=growth_stage_labels.index(
                get_default_label(choices["growth_stage"], defaults["growth_stage"])
            ),
        )
        soil_moisture = col4.slider(
            "Soil moisture",
            min_value=int(stats["soil_moisture"]["min"]),
            max_value=int(stats["soil_moisture"]["max"]),
            value=int(round(stats["soil_moisture"]["default"])),
        )

        col5, col6, col7 = st.columns(3)
        temperature = col5.number_input(
            "Temperature",
            min_value=float(stats["Temperature"]["min"]),
            max_value=float(stats["Temperature"]["max"]),
            value=float(round(stats["Temperature"]["default"], 2)),
            step=0.1,
        )
        humidity = col6.number_input(
            "Humidity",
            min_value=float(stats["Humidity"]["min"]),
            max_value=float(stats["Humidity"]["max"]),
            value=float(round(stats["Humidity"]["default"], 2)),
            step=0.1,
        )
        rainfall = col7.number_input(
            "Rainfall",
            min_value=float(stats["Rainfall"]["min"]),
            max_value=float(stats["Rainfall"]["max"]),
            value=float(round(stats["Rainfall"]["default"], 2)),
            step=0.1,
        )

        submitted = st.form_submit_button("Check irrigation need", use_container_width=True)

    if not submitted:
        return

    crop_value = choices["Crop"][crop_label]
    soil_type_value = choices["soil_type"][soil_type_label]
    growth_stage_value = choices["growth_stage"][growth_stage_label]

    water_feature = estimate_water_feature(
        soil_moisture=soil_moisture,
        rainfall=rainfall,
        temperature=temperature,
        soil_type=soil_type_value,
        growth_stage=growth_stage_value,
    )

    input_frame = pd.DataFrame(
        [
            {
                "Temperature": temperature,
                "Humidity": humidity,
                "Rainfall": rainfall,
                "Crop": crop_value,
                "soil_moisture": soil_moisture,
                "soil_type": soil_type_value,
                "growth_stage": growth_stage_value,
                "water_required_mm": water_feature,
            }
        ]
    )

    transformed_input = artifacts["preprocessor"].transform(input_frame)
    prediction = int(model.predict(transformed_input)[0])

    irrigation_probability = None
    if hasattr(model, "predict_proba"):
        irrigation_probability = float(model.predict_proba(transformed_input)[0][1])

    water_plan = estimate_irrigation_amount(
        soil_type=soil_type_value,
        temperature=temperature,
        growth_stage=growth_stage_value,
    )

    if prediction == 1:
        st.warning("Irrigation is recommended for the current conditions.")
    else:
        st.success("No irrigation is recommended right now.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Soil moisture", f"{soil_moisture}%")
    metric_columns[1].metric(
        "Irrigation probability",
        f"{irrigation_probability:.1%}" if irrigation_probability is not None else "n/a",
    )
    metric_columns[2].metric(
        "Estimated water plan",
        f"{water_plan} mm" if prediction == 1 else "0 mm",
    )

    st.caption(
        "The irrigation model was trained with a derived water feature, so the app "
        f"rebuilds that notebook logic internally. Current derived value: {water_feature} mm."
    )

    st.dataframe(input_frame, use_container_width=True, hide_index=True)


def render_yield_page(model: object, artifacts: dict[str, object]) -> None:
    st.subheader("Yield Prediction")
    st.write("Forecast crop yield using the production statistics model inputs.")

    stats = artifacts["stats"]
    choices = artifacts["choices"]
    defaults = artifacts["defaults"]

    state_labels = list(choices["State"].keys())
    crop_labels = list(choices["Crop"].keys())
    season_labels = list(choices["Season"].keys())

    with st.form("yield_prediction_form"):
        col1, col2 = st.columns(2)
        state_label = col1.selectbox(
            "State",
            state_labels,
            index=state_labels.index(get_default_label(choices["State"], defaults["State"])),
        )
        crop_label = col2.selectbox(
            "Crop",
            crop_labels,
            index=crop_labels.index(get_default_label(choices["Crop"], defaults["Crop"])),
        )

        col3, col4, col5 = st.columns(3)
        season_label = col3.selectbox(
            "Season",
            season_labels,
            index=season_labels.index(get_default_label(choices["Season"], defaults["Season"])),
        )
        crop_year = col4.number_input(
            "Crop year",
            min_value=int(stats["Crop_Year"]["min"]),
            max_value=int(stats["Crop_Year"]["max"]),
            value=int(stats["Crop_Year"]["default"]),
            step=1,
        )
        area = col5.number_input(
            "Area",
            min_value=float(max(0.0, stats["Area"]["min"])),
            max_value=float(stats["Area"]["max"]),
            value=float(round(stats["Area"]["default"], 2)),
            step=1.0,
        )

        submitted = st.form_submit_button("Predict yield", use_container_width=True)

    if not submitted:
        return

    input_frame = pd.DataFrame(
        [
            {
                "Crop_Year": crop_year,
                "Area": area,
                "Crop": choices["Crop"][crop_label],
                "State": choices["State"][state_label],
                "Season": choices["Season"][season_label],
            }
        ]
    )

    transformed_input = artifacts["preprocessor"].transform(input_frame)
    predicted_yield = float(model.predict(transformed_input)[0])

    st.success(f"Predicted yield: {predicted_yield:.2f}")

    summary_columns = st.columns(2)
    summary_columns[0].metric("Crop year", f"{crop_year}")
    summary_columns[1].metric("Area", f"{area:.2f}")

    st.dataframe(input_frame, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Smart Agriculture Control Center",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_app_style()

    models = load_models()
    crop_artifacts = prepare_crop_artifacts()
    irrigation_artifacts = prepare_irrigation_artifacts()
    yield_artifacts = prepare_yield_artifacts()

    render_header()

    st.sidebar.title("Navigation")
    st.sidebar.caption("Run locally with: streamlit run app.py")
    selected_page = st.sidebar.radio(
        "Choose a prediction task",
        ["Crop Recommendation", "Irrigation Recommendation", "Yield Prediction"],
    )

    if selected_page == "Crop Recommendation":
        render_crop_page(models["crop"], crop_artifacts)
    elif selected_page == "Irrigation Recommendation":
        render_irrigation_page(models["irrigation"], irrigation_artifacts)
    else:
        render_yield_page(models["yield"], yield_artifacts)


if __name__ == "__main__":
    main()
