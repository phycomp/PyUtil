streamlit pandas DataFrame data_editor 進階範例
The data_editor widget in Streamlit allows users to interactively edit a pandas DataFrame directly within the app, making it a powerful tool for dynamic data manipulation and analysis. Here’s an advanced use case that showcases how to combine data_editor with other Streamlit components for data filtering, validation, and exporting edited data.
Features of this Example:
    1. Advanced Filtering and Data Validation: Enable filtering and validation on user inputs.
    2. Automatic Summary Calculation: Calculate and display real-time summaries (e.g., sum, mean) based on user-edited data.
    3. Exporting Data: Allow users to download the edited data in CSV format.
    4. Handling Conditional Formatting: Highlight specific values or rows based on conditions.
Example Code:
import streamlit as st
import pandas as pd

# Sample DataFrame
data = {
    "Product": ["A", "B", "C", "D"],
    "Price": [100, 200, 150, 250],
    "Quantity": [30, 40, 20, 50],
    "Category": ["Electronics", "Furniture", "Electronics", "Furniture"]
}
df = pd.DataFrame(data)

# Title of the Streamlit App
st.title("Advanced Data Editing with Streamlit and Pandas")

# Filter Sidebar
st.sidebar.header("Filter Options")

# Category Filter
category_filter = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

# Price Filter (slider)
min_price, max_price = st.sidebar.slider("Select Price Range", int(df["Price"].min()), int(df["Price"].max()), (100, 250))

# Apply the filters to the DataFrame
filtered_df = df[(df["Category"].isin(category_filter)) & (df["Price"].between(min_price, max_price))]

# Display DataFrame in the data editor for user to edit
edited_df = st.data_editor(filtered_df, num_rows="dynamic")

# Show Summary Metrics based on the edited data
st.subheader("Summary of Edited Data")

# Total Price (sum)
total_price = edited_df["Price"].sum()
st.metric(label="Total Price", value=total_price)

# Average Price
average_price = edited_df["Price"].mean()
st.metric(label="Average Price", value=average_price)

# Total Quantity
total_quantity = edited_df["Quantity"].sum()
st.metric(label="Total Quantity", value=total_quantity)

# Validate the Edited Data
st.subheader("Validation")
invalid_data = edited_df[edited_df["Price"] <= 0]  # Example of invalid condition
if not invalid_data.empty:
    st.error("Some rows have invalid prices (less than or equal to zero). Please fix the errors.")
    st.write(invalid_data)

# Allow users to export the edited DataFrame as a CSV
st.subheader("Download Edited Data")
csv = edited_df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", data=csv, file_name="edited_data.csv", mime="text/csv")

# Conditional formatting (highlight rows with specific condition)
st.subheader("Highlighted Data")
highlighted_df = edited_df.style.apply(
    lambda x: ["background-color: yellow" if v > 200 else "" for v in x["Price"]], axis=1
)
st.dataframe(highlighted_df)
Explanation of Key Features:
1. Filtering Data Before Editing
In the sidebar, users can filter the data by Category or by a price range using a slider. The filtered data is then displayed in the data_editor for interactive editing.
category_filter = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())
min_price, max_price = st.sidebar.slider("Select Price Range", int(df["Price"].min()), int(df["Price"].max()), (100, 250))
filtered_df = df[(df["Category"].isin(category_filter)) & (df["Price"].between(min_price, max_price))]
2. Real-Time Summary Calculation
After editing the data, the app calculates key metrics like the total and average price, and total quantity, updating dynamically as the user makes changes in the data_editor.
total_price = edited_df["Price"].sum()
average_price = edited_df["Price"].mean()
total_quantity = edited_df["Quantity"].sum()
These metrics are displayed using st.metric to give users a quick view of the dataset's summary statistics.
3. Data Validation
The app performs a basic data validation step to check if any rows have invalid values (e.g., price less than or equal to zero). If any invalid data is detected, the app highlights the invalid rows and shows an error message.
invalid_data = edited_df[edited_df["Price"] <= 0]
if not invalid_data.empty:
    st.error("Some rows have invalid prices (less than or equal to zero). Please fix the errors.")
    st.write(invalid_data)
4. Exporting Edited Data
After editing the data, users can export the modified DataFrame as a CSV file. The st.download_button widget is used to allow users to download the edited data.
csv = edited_df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", data=csv, file_name="edited_data.csv", mime="text/csv")
5. Conditional Formatting
To highlight specific values in the DataFrame (e.g., prices greater than 200), the app uses pandas' styling capabilities. Here, rows where the price is above 200 are highlighted with a yellow background.
highlighted_df = edited_df.style.apply(
    lambda x: ["background-color: yellow" if v > 200 else "" for v in x["Price"]], axis=1
)
st.dataframe(highlighted_df)
Customization Ideas:
    • Validation Rules: Add more sophisticated validation rules (e.g., checking for missing values, or ensuring values in one column are greater than in another).
    • Conditional Formatting: Apply more complex conditional formatting to highlight specific rows or cells based on multiple conditions.
    • Real-Time Analytics: Add charts or visualizations (e.g., bar charts, line charts) that update in real-time based on the edited data.
    • Column Data Types: Set specific column data types and validations directly inside the data_editor (for example, making sure certain columns only accept integers or floats).
This example showcases how you can use the data_editor to create an interactive, user-friendly data-editing experience while integrating advanced features like filtering, validation, real-time analysis, and conditional formatting into your Streamlit app!
