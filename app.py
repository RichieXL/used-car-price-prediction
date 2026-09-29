import streamlit as st
import pandas as pd
import joblib

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_car_price_model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Car Price Prediction MVP")

st.write(
    "Enter vehicle information to estimate the selling price."
)


# ============================================================
# VEHICLE INFORMATION
# ============================================================

st.header("Vehicle Information")

# Vehicle Year dropdown
year_options = list(range(1990, 2027))

year = st.selectbox(
    "Vehicle Year",
    year_options,
    index=year_options.index(2012)
)


# Make dropdown
make_options = [
    "Ford",
    "Chevrolet",
    "Toyota",
    "Honda",
    "Nissan",
    "Hyundai",
    "Kia",
    "Jeep",
    "Ram",
    "GMC",
    "Dodge",
    "Subaru",
    "Volkswagen",
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "Lexus",
    "Mazda",
    "Buick",
    "Cadillac",
    "Chrysler",
    "Volvo",
    "Acura",
    "Infiniti",
    "Lincoln",
    "Mitsubishi",
    "Pontiac",
    "Saturn",
    "Other"
]

make = st.selectbox(
    "Make",
    make_options,
    index=make_options.index("Ford")
)


# Model dropdown
model_options = [
    "F-150",
    "Silverado",
    "Camry",
    "Corolla",
    "Civic",
    "Accord",
    "CR-V",
    "RAV4",
    "Altima",
    "Malibu",
    "Equinox",
    "Tucson",
    "Elantra",
    "Sonata",
    "Wrangler",
    "Grand Cherokee",
    "Escape",
    "Explorer",
    "Fusion",
    "Mustang",
    "Other"
]

model_name = st.selectbox(
    "Model",
    model_options,
    index=model_options.index("F-150")
)


# Trim dropdown
trim_options = [
    "Base",
    "SE",
    "SEL",
    "LE",
    "XLE",
    "XLT",
    "Lariat",
    "Limited",
    "Sport",
    "Touring",
    "EX",
    "LX",
    "S",
    "SV",
    "SL",
    "LT",
    "LTZ",
    "LS",
    "Premium",
    "Other"
]

trim = st.selectbox(
    "Trim",
    trim_options,
    index=trim_options.index("Base")
)


# Body type dropdown
body_options = [
    "Sedan",
    "SUV",
    "Pickup",
    "Coupe",
    "Hatchback",
    "Wagon",
    "Convertible",
    "Van",
    "Minivan",
    "Truck",
    "Other"
]

body = st.selectbox(
    "Body Type",
    body_options,
    index=body_options.index("Sedan")
)


# ============================================================
# VEHICLE CONDITION / PRICING
# ============================================================

st.header("Vehicle Details")

odometer = st.number_input(
    "Odometer",
    min_value=0,
    max_value=1_000_000,
    value=50_000,
    step=1_000
)

market_price = st.number_input(
    "Market Price",
    min_value=0,
    value=12_000,
    step=500
)


# ============================================================
# TRANSMISSION / APPEARANCE
# ============================================================

transmission = st.selectbox(
    "Transmission",
    [
        "automatic",
        "manual"
    ]
)


color_options = [
    "White",
    "Black",
    "Silver",
    "Gray",
    "Red",
    "Blue",
    "Green",
    "Brown",
    "Gold",
    "Orange",
    "Yellow",
    "Purple",
    "Other"
]

color = st.selectbox(
    "Exterior Color",
    color_options,
    index=color_options.index("White")
)


interior_options = [
    "Black",
    "Gray",
    "Beige",
    "Tan",
    "Brown",
    "Red",
    "Other"
]

interior = st.selectbox(
    "Interior Color",
    interior_options,
    index=interior_options.index("Black")
)


# ============================================================
# LOCATION
# ============================================================

state_options = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"
]

state = st.selectbox(
    "State",
    state_options,
    index=state_options.index("Ohio")
)


# ============================================================
# SALE INFORMATION
# ============================================================

st.header("Sale Information")

sale_year_options = list(range(2014, 2027))

sale_year = st.selectbox(
    "Sale Year",
    sale_year_options,
    index=sale_year_options.index(2015)
)


sale_month_options = list(range(1, 13))

sale_month = st.selectbox(
    "Sale Month",
    sale_month_options,
    index=sale_month_options.index(6)
)


sale_day_options = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

sale_day = st.selectbox(
    "Sale Day of Week",
    options=list(sale_day_options.keys()),
    format_func=lambda x: sale_day_options[x],
    index=2
)


# ============================================================
# DERIVED FEATURE
# ============================================================

# Calculate vehicle age automatically
vehicle_age = sale_year - year


st.write(f"Vehicle Age: **{vehicle_age} years**")


# ============================================================
# CREATE MODEL INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame([{

    "Year": year,

    "Odometer": odometer,

    "Market Price": market_price,

    "Sale Year": sale_year,

    "Sale Month": sale_month,

    "Sale Day of Week": sale_day,

    "Vehicle Age": vehicle_age,

    "Make": make,

    "Model": model_name,

    "Trim": trim,

    "Body": body,

    "Transmission": transmission,

    "Color": color,

    "Interior": interior,

    "State": state,

}])


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Selling Price",
    type="primary"
):

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Selling Price: ${prediction:,.2f}"
    )

    # Display the information used for prediction
    st.subheader("Prediction Details")

    st.write(
        f"**Vehicle:** {year} {make} {model_name} {trim}"
    )

    st.write(
        f"**Body:** {body}"
    )

    st.write(
        f"**Odometer:** {odometer:,} miles"
    )

    st.write(
        f"**State:** {state}"
    )
# uv run streamlit run app.py