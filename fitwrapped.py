import sqlite3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import time
import geopandas as gpd
from shapely.geometry import Point
import us
import plotly.express as px

class FitWrapped:
    def __init__(self, year=2025):
        self.console = Console()
        self.db_path = "DBs"
        self.rhr_path = "RHR"
        self.sleep_path = "Sleep"
        self.year = year  # Based on the data files we saw
        
        # Connect to Garmin databases
        self.garmin_db = sqlite3.connect(os.path.join(self.db_path, "garmin.db"))
        self.activities_db = sqlite3.connect(os.path.join(self.db_path, "garmin_activities.db"))
        self.monitoring_db = sqlite3.connect(os.path.join(self.db_path, "garmin_monitoring.db"))
        
        # Load RHR and Sleep data
        self.rhr_data = self._load_rhr_data()
        self.sleep_data = self._load_sleep_data()
        self.run_sleep_data = self._get_run_sleep()
        self.location_data = self._get_coords()

    def _load_rhr_data(self):
        """Load and process RHR data from JSON files"""
        rhr_data = []
        rhr_files = Path(self.rhr_path).glob(f"rhr_{self.year}-*.json")
        
        for file in rhr_files:
            with open(file) as f:
                data = json.load(f)
                metrics = data.get("allMetrics", {}).get("metricsMap", {}).get("WELLNESS_RESTING_HEART_RATE", [])
                if metrics:
                    entry = metrics[0]
                    rhr_data.append({
                        "date": entry.get("calendarDate", file.stem.split("_")[1]),
                        "value": entry.get("value")
                    })

        df = pd.DataFrame(rhr_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    def _get_coords(self):
        query = "SELECT * from steps_activities_view WHERE strftime('%Y', start_time) = ?"
        df = pd.read_sql_query(query, self.activities_db,  params=(str(self.year),))
        states = gpd.read_file("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json")

        def extract_coords(url):
            if not isinstance(url, str):
                return None, None
            match = re.search(r'q=([\-\d\.]+),([\-\d\.]+)', url)
            if match:
                return float(match.group(1)), float(match.group(2))
            return None, None
        
        df['lat'], df['lon'] = zip(*df['start_loc'].map(extract_coords))
        df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        states = states[['NAME', 'geometry']]

        # Join: assign each run to a state polygon if it's within one
        gdf_with_state = gpd.sjoin(gdf, states, how='left', predicate='within')

        # Rename the column for clarity
        gdf_with_state = gdf_with_state.rename(columns={'NAME': 'state'})
        
        state_counts = gdf_with_state['state'].value_counts().reset_index()
        state_counts.columns = ['state', 'count']
        return state_counts
                
    def _load_sleep_data(self):
        """Load and process sleep data from JSON files"""
        sleep_data = []
        sleep_files = Path(self.sleep_path).glob(f"sleep_{self.year}-*.json")
        
        for file in sleep_files:
            with open(file) as f:
                data = json.load(f)
                dto = data.get("dailySleepDTO")
                if dto:
                    dto['date'] = file.stem.split('_')[1]  # Add date for timeline plots
                    sleep_data.append(dto)
        df = pd.DataFrame(sleep_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['sleep_hours'] = df['deepSleepSeconds'] / 3600
            df['rolling_sleep'] = df['sleep_hours'].rolling(window=10, min_periods=1).mean()
        return df

    def _get_activity_breakdown(self):
        """Get detailed analytics per activity type"""
        query = """
        SELECT 
            sport AS activity_type,
            COUNT(*) AS count,
            SUM(distance) AS total_distance,
            AVG(distance) AS avg_distance,
            SUM(CAST(strftime('%H', elapsed_time) AS INTEGER) * 60 +
            CAST(strftime('%M', elapsed_time) AS INTEGER) +
            CAST(strftime('%S', elapsed_time) AS INTEGER) / 60.0) AS total_duration_minutes,
            AVG(CAST(strftime('%H', elapsed_time) AS INTEGER) * 60 +
            CAST(strftime('%M', elapsed_time) AS INTEGER) +
            CAST(strftime('%S', elapsed_time) AS INTEGER) / 60.0) as avg_duration_minutes,
            AVG(avg_hr) AS avg_heart_rate
        FROM activities
        WHERE strftime('%Y', start_time) = ?
        GROUP BY sport
        ORDER BY count DESC
        """
        df = pd.read_sql_query(query, self.activities_db, params=(str(self.year),))
        return df
    
    def _get_best_activities(self, top_n=5):
        """Fetch top activities by distance"""
        query = """
        SELECT
            name,
            sport,
            ROUND(distance,1) AS distance,
            (CAST(strftime('%H', elapsed_time) AS INTEGER) * 60 +
             CAST(strftime('%M', elapsed_time) AS INTEGER) +
             CAST(strftime('%S', elapsed_time) AS INTEGER) / 60.0) AS duration_minutes,
            calories
        FROM activities
        WHERE strftime('%Y', start_time) = ?
        ORDER BY distance DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, self.activities_db, params=(str(self.year), top_n))
        return df

    def _get_run_sleep(self):
        run_times_df = pd.read_sql_query("""
        SELECT start_time
        FROM activities
        WHERE sport = 'running' AND strftime('%Y', start_time) = ?
        """, self.activities_db, params=(str(self.year),))

        run_times_df['start_time'] = pd.to_datetime(run_times_df['start_time'])
        run_times_df['date'] = run_times_df['start_time'].dt.date
        run_times_df['hour'] = run_times_df['start_time'].dt.hour + run_times_df['start_time'].dt.minute / 60
        return run_times_df

    def _get_sleep_insights(self):
        if self.sleep_data.empty:
            return {}
        sleep = self.sleep_data
        return {
            'average_sleep_duration': sleep['sleep_hours'].mean(),
            'best_sleep_date': sleep.loc[sleep['sleep_hours'].idxmax(), 'date'].date(),
            'total_sleep_hours': sleep['sleep_hours'].sum()
        }

    def _get_heart_rate_insights(self):
        if self.rhr_data.empty:
            return {}
        rhr = self.rhr_data
        return {
            'average_rhr': rhr['value'].mean(),
            'lowest_rhr': rhr['value'].min(),
            'lowest_rhr_date': rhr.loc[rhr['value'].idxmin(), 'date'].date()
        }

    def generate_wrapped(self):
        self.console.print("\n[bold magenta]🎉 Welcome to FitWrapped 2025! 🎉[/bold magenta]\n", justify="center")
        breakdown = self._get_activity_breakdown()
        sleep_insights = self._get_sleep_insights()
        heart_rate_insights = self._get_heart_rate_insights()
        best_activities = self._get_best_activities()
        total_activities = breakdown['count'].sum()
        most_common = breakdown.iloc[0]['activity_type'] if not breakdown.empty else 'N/A'

        panel_text = Text.from_markup(f"""
[bold green]Year in Fitness[/bold green]
🏃 Total Activities: {total_activities}
⭐ Most Common: {most_common}
💪 Active Hours: {breakdown['total_duration_minutes'].sum()/60:.1f}

[bold blue]Sleep & Recovery[/bold blue]
😴 Avg Deep Sleep: {sleep_insights.get('average_sleep_duration', 0):.1f} hrs
🌟 Best Sleep: {sleep_insights.get('best_sleep_date', 'N/A')}

[bold red]Heart Rate[/bold red]
❤️ Avg RHR: {heart_rate_insights.get('average_rhr', 0):.1f} bpm
💓 Lowest RHR: {heart_rate_insights.get('lowest_rhr_date', 'N/A')} @ {heart_rate_insights.get('lowest_rhr', 0):.0f} bpm
""", justify="left")
        self.console.print(Panel(panel_text, title="Your Fitness Summary", box=box.ROUNDED, padding=(1,2)))

        if not breakdown.empty:
            table = Table(title="Activity Breakdown", box=box.SIMPLE_HEAVY)
            for col, header in [
                ('activity_type', 'Activity'), ('count', 'Count'),
                ('total_distance', 'Total Dist (mi)'), ('avg_distance', 'Avg Dist (mi)'),
                ('total_duration_minutes', 'Total Dur (h)'), ('avg_duration_minutes', 'Avg Dur (min)'),
                ('avg_heart_rate', 'Avg HR')]:
                table.add_column(header, justify="right" if col != 'activity_type' else "left")

            for _, row in breakdown.iterrows():
                table.add_row(
                    row['activity_type'],
                    str(int(row['count'])),
                    f"{row['total_distance']:.1f}",
                    f"{row['avg_distance']:.1f}",
                    f"{row['total_duration_minutes']/60:.1f}",
                    f"{row['avg_duration_minutes']:.1f}",
                    f"{row['avg_heart_rate']:.0f}"
                )
            self.console.print(table)

        self._create_dashboard(breakdown, best_activities)
        self.console.print("\nDashboard saved as [bold]fitness_dashboard.html[/bold]")

    def _create_dashboard(self, breakdown, best_activities):
        fig = make_subplots(
            rows=4, cols=2,
            specs=[
                [{}, {}],
                [{}, {}],
                [{}, {'type': 'choropleth'}],
                [{'type': 'table', 'colspan': 2}, None]
            ],
            subplot_titles=(
                "Activity Counts", "Activity Distance",
                "Sleep Trends", "Resting HR Trends",
                "Running Times", "State Distribution",
                "Best Activities", ""
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        # Row 1
        fig.add_trace(go.Bar(x=breakdown['activity_type'], y=breakdown['count'], name='Count'), row=1, col=1)
        fig.add_trace(go.Bar(x=breakdown['activity_type'], y=breakdown['total_distance'], name='Total Dist (mi)'), row=1, col=2)
        # Row 2
        if not self.sleep_data.empty:
            sleep_df = self.sleep_data.sort_values('date')
            fig.add_trace(go.Scatter(x=sleep_df['date'], y=sleep_df['rolling_sleep'], mode='lines', name='10d Avg Sleep'), row=2, col=1)
            avg_sleep = sleep_df['sleep_hours'].mean()
            fig.add_hline(y=avg_sleep, line_dash='dash', annotation_text=f"Avg {avg_sleep:.1f}h", row=2, col=1)
        if not self.rhr_data.empty:
            rhr_df = self.rhr_data.sort_values('date')
            rhr_df['rolling'] = rhr_df['value'].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=rhr_df['date'], y=rhr_df['value'], mode='markers', name='Daily RHR'), row=2, col=2)
            fig.add_trace(go.Scatter(x=rhr_df['date'], y=rhr_df['rolling'], mode='lines', name='7d Avg RHR'), row=2, col=2)
            avg_r = rhr_df['value'].mean()
            fig.add_hline(y=avg_r, line_dash='dash', annotation_text=f"Avg {avg_r:.0f} bpm", row=2, col=2)
        # Row 3
        if not self.run_sleep_data.empty:
            fig.add_trace(go.Scatter(
                x=self.run_sleep_data['date'],
                y=self.run_sleep_data['hour'],
                mode='markers',
                marker=dict(size=8),
                name='Running Time'
            ), row=3, col=1)
            fig.update_yaxes(title_text='Hour of Day', range=[0,24], row=3, col=1)
        if not self.location_data.empty:
            loc = self.location_data.copy()
            loc['state'] = loc['state'].map(lambda x: us.states.lookup(x).abbr if pd.notna(x) else None)
            all_states = pd.DataFrame({'state':[s.abbr for s in us.states.STATES]})
            full_data = all_states.merge(loc, on='state', how='left').fillna(0)
            choropleth = go.Choropleth(
                locations=full_data['state'], locationmode='USA-states', z=full_data['count'], showscale=False
            )
            fig.add_trace(choropleth, row=3, col=2)
            fig.update_geos(scope='usa', showlakes=True, lakecolor='lightgray', bgcolor='rgba(0,0,0,0)')
        # Row 4: Best Activities Table
        if not best_activities.empty:
            fig.add_trace(go.Table(
                header=dict(values=["Name","Sport","Distance (mi)","Duration (min)","Calories"]),
                cells=dict(values=[
                    best_activities['name'], best_activities['sport'],
                    best_activities['distance'], best_activities['duration_minutes'], best_activities['calories']
                ])
            ), row=4, col=1)
        # Layout
        fig.update_layout(
            title_text=f"FitWrapped {self.year} Dashboard",
            template='plotly_white', height=1000, width=1250,
            margin=dict(l=50, r=50, t=100, b=50),
            title_font=dict(size=24), font=dict(size=14),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        fig.update_xaxes(tickangle=-45)
        fig.write_html("fitness_dashboard.html")

if __name__ == "__main__":
    wrapped = FitWrapped()
    wrapped.generate_wrapped()