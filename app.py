from pathlib import Path
import os
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Handle deployment environment paths
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

# Comprehensive directory scanning for deployment
def find_data_and_model_dirs():
    """Find data and model directories by scanning the filesystem"""
    
    # Start from current working directory and scan up
    current_dir = Path.cwd()
    search_paths = []
    
    # Add current directory and parents
    temp_dir = current_dir
    for _ in range(5):  # Go up max 5 levels
        search_paths.extend([
            temp_dir,
            temp_dir / "data",
            temp_dir / "models",
            temp_dir / "src",
            temp_dir / "app"
        ])
        temp_dir = temp_dir.parent
    
    # Add common deployment paths
    search_paths.extend([
        Path("/mount/src/smart-agri"),
        Path("/app"),
        Path("/workspace"),
        Path("/home/app"),
        BASE_DIR
    ])
    
    # Look for model files and data files
    model_files_found = []
    data_files_found = []
    
    for path in search_paths:
        if path.exists() and path.is_dir():
            # Check for model files
            for model_file in ["crop_recommendation_model.pkl", "irrigation_model.pkl", "yield_prediction_model.pkl"]:
                if (path / model_file).exists():
                    model_files_found.append(path)
                    break
            
            # Check for data files
            for data_file in ["Crop_Recommendation.csv", "irrigation_recommendation_dataset.csv", "indian crop production.csv"]:
                if (path / data_file).exists():
                    data_files_found.append(path)
                    break
    
    # Find common parent directories
    model_dir = None
    data_dir = None
    
    if model_files_found:
        # Find the directory that contains all model files
        for path in model_files_found:
            if all((path / f).exists() for f in ["crop_recommendation_model.pkl", "irrigation_model.pkl", "yield_prediction_model.pkl"]):
                model_dir = path
                break
    
    if data_files_found:
        # Find the directory that contains all data files
        for path in data_files_found:
            if all((path / f).exists() for f in ["Crop_Recommendation.csv", "irrigation_recommendation_dataset.csv", "indian crop production.csv"]):
                data_dir = path
                break
    
    return data_dir, model_dir

DATA_DIR, MODEL_DIR = find_data_and_model_dirs()

# Fallback to original paths if not found
if DATA_DIR is None:
    DATA_DIR = BASE_DIR / "data"
if MODEL_DIR is None:
    MODEL_DIR = BASE_DIR / "models"

# Debug information for deployment
if __name__ != "__main__":
    st.write(f"Debug: Working directory: {Path.cwd()}")
    st.write(f"Debug: BASE_DIR: {BASE_DIR}")
    st.write(f"Debug: DATA_DIR: {DATA_DIR}")
    st.write(f"Debug: MODEL_DIR: {MODEL_DIR}")
    st.write(f"Debug: Data dir exists: {DATA_DIR.exists()}")
    st.write(f"Debug: Model dir exists: {MODEL_DIR.exists()}")
    
    # Show files in current directory
    try:
        current_files = list(Path.cwd().iterdir())
        st.write(f"Debug: Files in current directory: {[f.name for f in current_files[:10]]}")
    except:
        st.write("Debug: Could not list current directory files")
    
    # Show parent directory structure
    try:
        parent = Path.cwd().parent
        if parent.exists():
            parent_files = list(parent.iterdir())
            st.write(f"Debug: Files in parent directory: {[f.name for f in parent_files[:10]]}")
    except:
        st.write("Debug: Could not list parent directory files")


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
    try:
        models = {}
        model_files = {
            "crop": "crop_recommendation_model.pkl",
            "irrigation": "irrigation_model.pkl", 
            "yield": "yield_prediction_model.pkl"
        }
        
        # List all files in model directory for debugging
        if MODEL_DIR.exists():
            model_files_list = list(MODEL_DIR.glob("*.pkl"))
            st.write(f"Debug: Found model files: {[f.name for f in model_files_list]}")
        else:
            st.error(f"Model directory does not exist: {MODEL_DIR}")
            st.stop()
        
        for model_name, filename in model_files.items():
            model_path = MODEL_DIR / filename
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                st.error(f"Available files in {MODEL_DIR}: {list(MODEL_DIR.iterdir()) if MODEL_DIR.exists() else 'Directory not found'}")
                st.stop()
            models[model_name] = joblib.load(model_path)
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error(f"MODEL_DIR: {MODEL_DIR}")
        st.error(f"Working directory: {Path.cwd()}")
        st.stop()


@st.cache_resource(show_spinner=False)
def prepare_crop_artifacts() -> dict[str, object]:
    try:
        crop_data_path = DATA_DIR / "Crop_Recommendation.csv"
        if not crop_data_path.exists():
            st.error(f"Crop data file not found: {crop_data_path}")
            st.stop()
        dataframe = pd.read_csv(crop_data_path)
        
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
    except Exception as e:
        st.error(f"Error preparing crop artifacts: {str(e)}")
        st.stop()


@st.cache_resource(show_spinner=False)
def prepare_irrigation_artifacts() -> dict[str, object]:
    try:
        irrigation_data_path = DATA_DIR / "irrigation_recommendation_dataset.csv"
        if not irrigation_data_path.exists():
            st.error(f"Irrigation data file not found: {irrigation_data_path}")
            st.stop()
        dataframe = pd.read_csv(irrigation_data_path)

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

        # Define all possible categories that the model was trained on
        all_crop_categories = [
            'Apple', 'Banana', 'Blackgram', 'ChickPea', 'Grapes', 
            'KidneyBeans', 'Lentil', 'Maize', 'Mango', 'MothBeans', 
            'MungBean', 'Muskmelon', 'PigeonPeas', 'Pomegranate', 
            'Rice', 'Watermelon'
        ]
        
        all_soil_type_categories = ['Clay', 'Loamy', 'Sandy']
        all_growth_stage_categories = ['Early', 'Late', 'Mid']
        
        # Create categorical pipeline with all categories
        categorical_pipeline_fixed = Pipeline(
            steps=[
                ("imputation", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(categories=[all_crop_categories, all_soil_type_categories, all_growth_stage_categories], handle_unknown='ignore')),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_pipeline(), numeric_columns),
                ("categorical", categorical_pipeline_fixed, categorical_columns),
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
    except Exception as e:
        st.error(f"Error preparing irrigation artifacts: {str(e)}")
        st.stop()


@st.cache_resource(show_spinner=False)
def prepare_yield_artifacts() -> dict[str, object]:
    try:
        yield_data_path = DATA_DIR / "indian crop production.csv"
        if not yield_data_path.exists():
            st.error(f"Yield data file not found: {yield_data_path}")
            st.stop()
        dataframe = pd.read_csv(yield_data_path)
        
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

        # Define all possible categories that yield model was trained on
        all_yield_crops = [
            'Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Black pepper', 'Cashewnut', 
            'Castor seed', 'Coconut ', 'Coriander', 'Cotton(lint)', 'Cowpea(Lobia)', 
            'Dry chillies', 'Garlic', 'Ginger', 'Gram', 'Groundnut', 'Guar seed', 
            'Horse-gram', 'Jowar', 'Linseed', 'Maize', 'Masoor', 'Mesta', 
            'Moong(Green Gram)', 'Niger seed', 'Oilseeds total', 'Onion', 
            'Other Rabi pulses', 'Other Kharif pulses', 'Potato', 'Ragi', 
            'Rapeseed &Mustard', 'Rice', 'Safflower', 'Sesamum', 
            'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 
            'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'other oilseeds'
        ]
        
        all_yield_states = [
            'Andaman and Nicobar Island', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 
            'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 
            'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 
            'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'NCT of Delhi', 
            'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 
            'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 
            'West Bengal'
        ]
        
        all_yield_seasons = ['Autumn ', 'Kharif ', 'Rabi ', 'Summer ', 'Whole Year ']
        
        categorical_pipeline_yield = Pipeline(
            steps=[
                ("imputation", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(categories=[all_yield_crops, all_yield_states, all_yield_seasons], handle_unknown='ignore')),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_pipeline(), numeric_columns),
                ("categorical", categorical_pipeline_yield, categorical_columns),
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
    except Exception as e:
        st.error(f"Error preparing yield artifacts: {str(e)}")
        st.stop()


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

    # Debug information for feature mismatch
    st.write(f"Debug: Input features: {list(input_frame.columns)}")
    st.write(f"Debug: Input shape: {input_frame.shape}")
    
    transformed_input = artifacts["preprocessor"].transform(input_frame)
    st.write(f"Debug: Transformed shape: {transformed_input.shape}")
    
    # Get feature names from preprocessor
    try:
        feature_names = artifacts["preprocessor"].get_feature_names_out()
        st.write(f"Debug: Feature names: {list(feature_names)}")
    except:
        st.write("Debug: Could not get feature names")
    
    # Check model expected features
    if hasattr(model, 'n_features_in_'):
        st.write(f"Debug: Model expects {model.n_features_in_} features")
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'n_features_in_'):
        st.write(f"Debug: Model expects {model.best_estimator_.n_features_in_} features")
    
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

    # Debug information for feature mismatch
    st.write(f"Debug: Yield input features: {list(input_frame.columns)}")
    st.write(f"Debug: Yield input shape: {input_frame.shape}")
    
    transformed_input = artifacts["preprocessor"].transform(input_frame)
    st.write(f"Debug: Yield transformed shape: {transformed_input.shape}")
    
    # Get feature names from preprocessor
    try:
        feature_names = artifacts["preprocessor"].get_feature_names_out()
        st.write(f"Debug: Yield feature names: {list(feature_names)}")
    except:
        st.write("Debug: Could not get yield feature names")
    
    # Check model expected features
    if hasattr(model, 'n_features_in_'):
        st.write(f"Debug: Yield model expects {model.n_features_in_} features")
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'n_features_in_'):
        st.write(f"Debug: Yield model expects {model.best_estimator_.n_features_in_} features")
    
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
