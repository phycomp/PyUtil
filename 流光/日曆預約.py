import streamlit as st
from streamlit_calendar import calendar
import uuid
colors = ["blue", "red", "green"]

events = [
    {
        "title": "Event 1",
        "color": colors[2],
        "location": "LA",
        "start": "2024-08-30",
        "end": "2024-09-02",
        "resourceId": "b",
    }
]
people = [
    {"id": "a", "title": "Jeff"},
    {"id": "b", "title": "John"},
    {"id": "c", "title": "Both"}
]

calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "resources": people,
    "selectable": "true",
}

calendar_options = {
    **calendar_options,
    "headerToolbar": {
        "left": "today prev,next",
        "center": "title",
        "right": "dayGridDay,dayGridWeek,dayGridMonth",
    },
    "initialDate": "2024-08-01",
    "initialView": "dayGridMonth",
}
if not st.session_state.get("Calendar", False):
        st.session_state["Calendar"] = str(uuid.uuid4())

state = calendar(
    events=events,
    options=calendar_options,
    custom_css="""
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
    """,
    key=st.session_state["Calendar"],
)

event_to_add = {
    "title": "Event 7",
    "color": "#FF4B4B",
    "location": "SF",
    "start": "2024-09-01",
    "end": "2024-09-07",
    "resourceId": "a"
}

if st.button("add event"):
    events.append(event_to_add)
    st.write(events)
    st.session_state["Calendar"] = str(uuid.uuid4())
    st.rerun()

if state.get("eventsSet") is not None:
    st.session_state["events"] = state["eventsSet"]
