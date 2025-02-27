from streamlit import sidebar, slider, multiselect, data_editor, header, subheader, metric, error, dataframe, title, download_button, radio as stRadio, text_input, session_state
from pandas import DataFrame
from stUtil import rndrCode
MENU, 表單=[], ['dFrame', 'Validation', '下載CSV', '錯綜複雜', '二十四節氣']
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
  srch=text_input('搜尋', '')
if menu==len(表單):
  pass
elif menu==MENU[2]:
  #3. Data Validation The app performs a basic data validation step to check if any rows have invalid values (e.g., price less than or equal to zero). If any invalid data is detected, the app highlights the invalid rows and shows an error message.
  invalid_data = edited_df[edited_df["Price"] <= 0]
  if not invalid_data.empty:
      error("Some rows have invalid prices (less than or equal to zero). Please fix the errors.")
      rndrCode(invalid_data)
  #4. Exporting Edited Data After editing the data, users can export the modified DataFrame as a CSV file. The st.download_button widget is used to allow users to download the edited data.
  csv = edited_df.to_csv(index=False).encode("utf-8")
  download_button("Download as CSV", data=csv, file_name="edited_data.csv", mime="text/csv")
  #5. Conditional Formatting To highlight specific values in the DataFrame (e.g., prices greater than 200), the app uses pandas' styling capabilities. Here, rows where the price is above 200 are highlighted with a yellow background.
  highlighted_df = edited_df.style.apply(
      lambda x: ["background-color: yellow" if v > 200 else "" for v in x["Price"]], axis=1
  )
  dataframe(highlighted_df)
elif menu==MENU[1]:
  edited_df=session_state['dset']
  subheader("Validation") # Validate the Edited Data
  invalid_data = edited_df[edited_df["Price"] <= 0]  # Example of invalid condition
  if not invalid_data.empty:
      error("Some rows have invalid prices (less than or equal to zero). Please fix the errors.")
      rndrCode(invalid_data)

  # Allow users to export the edited DataFrame as a CSV
  subheader("Download Edited Data")
  csv = edited_df.to_csv(index=False).encode("utf-8")
  download_button("Download as CSV", data=csv, file_name="edited_data.csv", mime="text/csv")

  # Conditional formatting (highlight rows with specific condition)
  subheader("Highlighted Data")
  rndrCode(['Price=', edited_df['Price']])
  highlighted_df = edited_df.style.apply(   #for v in x
      lambda x: ["background-color: yellow" if x["Price"] > 200 else "" ], axis=1
  )
  dataframe(highlighted_df)
  #Explanation of Key Features: 1. Filtering Data Before Editing In the sidebar, users can filter the data by Category or by a price range using a slider. The filtered data is then displayed in the data_editor for interactive editing.
  with sidebar:
    category_filter = multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())
    min_price, max_price = slider("Select Price Range", int(df["Price"].min()), int(df["Price"].max()), (100, 250))
    filtered_df = df[(df["Category"].isin(category_filter)) & (df["Price"].between(min_price, max_price))]
  #2. Real-Time Summary Calculation After editing the data, the app calculates key metrics like the total and average price, and total quantity, updating dynamically as the user makes changes in the data_editor.
  total_price = edited_df["Price"].sum()
  average_price = edited_df["Price"].mean()
  total_quantity = edited_df["Quantity"].sum()

  #These metrics are displayed using st.metric to give users a quick view of the dataset's summary statistics.
elif menu==MENU[0]:
  data = { "Product": ["A", "B", "C", "D"], "Price": [100, 200, 150, 250], "Quantity": [30, 40, 20, 50], "Category": ["Electronics", "Furniture", "Electronics", "Furniture"] }
  df = DataFrame(data)

  title("Advanced Data Editing with Streamlit and Pandas") # Title of the Streamlit App

# Filter Sidebar

  with sidebar: # Category Filter
    header("Filter Options")
    category_filter = multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())
  min_price, max_price = slider("Select Price Range", int(df["Price"].min()), int(df["Price"].max()), (100, 250)) # Price Filter (slider)
  filtered_df = df[(df["Category"].isin(category_filter)) & (df["Price"].between(min_price, max_price))] # Apply the filters to the DataFrame
  edited_df=session_state['dset'] = data_editor(filtered_df, num_rows="dynamic") # Display DataFrame in the data editor for user to edit
  subheader("Summary of Edited Data") # Show Summary Metrics based on the edited data

  total_price = edited_df["Price"].sum() # Total Price (sum)
  metric(label="Total Price", value=total_price)

  average_price = edited_df["Price"].mean() # Average Price
  metric(label="Average Price", value=average_price)

  total_quantity = edited_df["Quantity"].sum() # Total Quantity
  metric(label="Total Quantity", value=total_quantity)
