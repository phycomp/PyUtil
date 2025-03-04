from streamlit import radio as stRadio, markdown as stMarkdown, dataframe, plotly_chart, multiselect
import numpy as np
from pandas import Dataframe, to_datetime, DatetimeIndex
import cufflinks

@st.cache
def get_data(url):
    df = pd.read_csv(url)
    df["date"] = to_datetime(df.date).dt.date
    df['date'] = DatetimeIndex(df.date)

    return df

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
data = get_data(url)

locations = data.location.unique().tolist()

with sidebar:
  analysis_type = stRadio("Analysis Type", ["Single", "Multiple"])
  stMarkdown(f"Analysis Mode: {analysis_type}")

  if analysis_type=="Single":
      location_selector = selectbox( "Select a Location", locations)
      stMarkdown(f"# Currently Selected {location_selector}")
      trend_level = selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
      stMarkdown(f"### Currently Selected {trend_level}")

      show_data = checkbox("Show Data")

      trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}
      trend_data = data.query(f"location=='{location_selector}'").\
          groupby(pd.Grouper(key="date", freq=trend_kwds[trend_level])).aggregate(new_cases=("new_cases", "sum"),
          new_deaths = ("new_deaths", "sum"),
          new_vaccinations = ("new_vaccinations", "sum"),
          new_tests = ("new_tests", "sum")).reset_index()

      trend_data["date"] = trend_data.date.dt.date

      new_cases = checkbox("New Cases")
      new_deaths = checkbox("New Deaths")
      new_vaccinations = checkbox("New Vaccinations")
      new_tests = checkbox("New Tests")

      lines = [new_cases, new_deaths, new_vaccinations, new_tests]
      line_cols = ["new_cases", "new_deaths", "new_vaccinations", "new_tests"]
      trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]

      if show_data:
          tcols = ["date"] + trends
          dataframe(trend_data[tcols])

      subplots=checkbox("Show Subplots", True)
      if len(trends)>0:
          fig=trend_data.iplot(kind="line", asFigure=True, xTitle="Date", yTitle="Values", x="date", y=trends, title=f"{trend_level} Trend of {', '.join(trends)}.", subplots=subplots)
          plotly_chart(fig, use_container_width=False)

  if analysis_type=="Multiple":
      selected = multiselect("Select Locations ", locations)
      st.markdown(f"## Selected Locations: {', '.join(selected)}")
      show_data = checkbox("Show Data")
      trend_level = selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
      stMarkdown(f"### Currently Selected {trend_level}")

      trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}

      trend_data = data.query(f"location in {selected}").\
          groupby(["location", pd.Grouper(key="date", 
          freq=trend_kwds[trend_level])]).aggregate(new_cases=("new_cases", "sum"),
          new_deaths = ("new_deaths", "sum"),
          new_vaccinations = ("new_vaccinations", "sum"),
          new_tests = ("new_tests", "sum")).reset_index()

      trend_data["date"] = trend_data.date.dt.date

      new_cases = checkbox("New Cases")
      new_deaths = checkbox("New Deaths")
      new_vaccinations = checkbox("New Vaccinations")
      new_tests = checkbox("New Tests")

      lines = [new_cases, new_deaths, new_vaccinations, new_tests]
      line_cols = ["new_cases", "new_deaths", "new_vaccinations", "new_tests"]
      trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]

      ndf = DataFrame(data=trend_data.date.unique(),columns=["date"])

      for s in selected:
          new_cols = ["date"]+[f"{s}_{c}" for c in line_cols]
          tdf = trend_data.query(f"location=='{s}'")
          tdf.drop("location", axis=1, inplace=True)
          tdf.columns=new_cols
          ndf=ndf.merge(tdf,on="date",how="inner")

      if show_data:
          if len(ndf)>0:
              ndf
              #dataframe(ndf)
          else:
              stMarkdown("Empty Dataframe")
      new_trends = []
      for c in trends:
          new_trends.extend([f"{s}_{c}" for s in selected])

      subplots=checkbox("Show Subplots", True)
      if len(trends)>0:
          stMarkdown("### Trend of Selected Locations")

          fig=ndf.iplot(kind="line", asFigure=True, xTitle="Date", yTitle="Values", x="date", y=new_trends, title=f"{trend_level} Trend of {', '.join(trends)}.", subplots=subplots)
          plotly_chart(fig, use_container_width=False)
