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
            match = re.search(r'q=([-\d\.]+),([-\d\.]+)', url)
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
    
    def _get_run_sleep(self):
        run_times_df = pd.read_sql_query("""
        SELECT start_time
        FROM activities
        WHERE sport = 'running' AND strftime('%Y', start_time) = ?
        """, self.activities_db, params=(str(self.year),))

        # Convert to datetime
        run_times_df['start_time'] = pd.to_datetime(run_times_df['start_time'])

        # Extract date and hour (for plotting)
        run_times_df['date'] = run_times_df['start_time'].dt.date
        run_times_df['hour'] = run_times_df['start_time'].dt.hour + run_times_df['start_time'].dt.minute / 60
        
        return run_times_df

    def _get_sleep_insights(self):
        """Analyze sleep patterns and quality"""
        sleep = self.sleep_data
        if sleep.empty:
            return {}
        sleep_stats = {
            'average_sleep_duration': sleep['sleep_hours'].mean(),
            'best_sleep_date': sleep.loc[sleep['sleep_hours'].idxmax(), 'date'].date(),
            'total_sleep_hours': sleep['sleep_hours'].sum()
        }
        return sleep_stats
    
    def _get_heart_rate_insights(self):
        """Analyze heart rate patterns"""
        rhr = self.rhr_data
        if rhr.empty:
            return {}
        rhr_stats = {
            'average_rhr': rhr['value'].mean(),
            'lowest_rhr': rhr['value'].min(),
            'lowest_rhr_date': rhr.loc[rhr['value'].idxmin(), 'date'].date()
        }
        return rhr_stats
    
    def generate_wrapped(self):
        """Generate and display the FitWrapped report"""
        self.console.print("\n[bold magenta]🎉 Welcome to FitWrapped 2025! 🎉[/bold magenta]\n", justify="center")
        
        # Pull analytics
        breakdown = self._get_activity_breakdown()
        sleep_insights = self._get_sleep_insights()
        heart_rate_insights = self._get_heart_rate_insights()
        total_activities = breakdown['count'].sum()
        most_common = breakdown.iloc[0]['activity_type'] if not breakdown.empty else 'N/A'
        
        # Summary Panel
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
        
        # Detailed Breakdown Table
        if not breakdown.empty:
            table = Table(title="Activity Breakdown", box=box.SIMPLE_HEAVY)
            for col, header in [
                ('activity_type', 'Activity'), ('count', 'Count'),
                ('total_distance', 'Total Dist (km)'), ('avg_distance', 'Avg Dist (km)'),
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
        
        # Generate dashboard HTML
        self._create_dashboard(breakdown)
        self.console.print("\nDashboard saved as [bold]fitness_dashboard.html[/bold]")

    def _create_dashboard(self, breakdown):
        """Create a dashboard with all visualizations"""
        # bump up spacing and give each row a little extra height
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{}, {}],         # row 1
                [{}, {}],         # row 2
                [{}, {'type': 'choropleth'}]  # row 3 — make (3,2) a choropleth cell
            ],
            subplot_titles=("Activity Counts", "Activity Distance", "Sleep Trends", "Resting HR Trends"),
            vertical_spacing=0.15,    # was 0.1
            horizontal_spacing=0.15,  # was 0.1
            row_heights=[0.3, 0.3, 0.3]
        )
        # Activity Count Bar
        fig.add_trace(go.Bar(x=breakdown['activity_type'], y=breakdown['count'], name='Count'), row=1, col=1)
        # Activity Distance Bar (km)
        fig.add_trace(go.Bar(x=breakdown['activity_type'], y=breakdown['total_distance'], name='Total Dist (km)'), row=1, col=2)
        
        # Sleep Trends
        if not self.sleep_data.empty:
            self.sleep_data = self.sleep_data.sort_values(by='date')
            fig.add_trace(go.Scatter(x=self.sleep_data['date'], y=self.sleep_data['rolling_sleep'], mode='lines', name='10d Avg Sleep'), row=2, col=1)
            avg = self.sleep_data['sleep_hours'].mean()
            fig.add_hline(y=avg, line_dash='dash', annotation_text=f"Avg {avg:.1f}h", row=2, col=1)
        
        # RHR Trends
        if not self.rhr_data.empty:
            self.rhr_data = self.rhr_data.sort_values('date')
            self.rhr_data['rolling'] = self.rhr_data['value'].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=self.rhr_data['date'], y=self.rhr_data['value'], mode='markers', name='Daily RHR'), row=2, col=2)
            fig.add_trace(go.Scatter(x=self.rhr_data['date'], y=self.rhr_data['rolling'], mode='lines', name='7d Avg RHR'), row=2, col=2)
            avg_r = self.rhr_data['value'].mean()
            fig.add_hline(y=avg_r, line_dash='dash', annotation_text=f"Avg {avg_r:.0f} bpm", row=2, col=2)
            
        if not self.run_sleep_data.empty:
            fig.add_trace(go.Scatter(
            x=self.run_sleep_data['date'], 
            y=self.run_sleep_data['hour'], 
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Running Time'
            ), row=3, col=1)

            fig.update_yaxes(title_text='Hour of Day (0–24)', range=[0, 24], row=3, col=1)
            fig.update_xaxes(title_text='Date', row=3, col=1)
            
        if not self.location_data.empty:
            self.location_data['state'] = self.location_data['state'].map(
            lambda x: us.states.lookup(x).abbr if pd.notna(x) and us.states.lookup(x) else None
            )            
            
            all_states = pd.DataFrame({
            'state': [s.abbr for s in us.states.STATES]
            })

            # Merge with your real counts
            full_data = all_states.merge(self.location_data, on='state', how='left')
            full_data['count'] = full_data['count'].fillna(0) 
            choropleth_trace = go.Choropleth(
                locations=full_data['state'],     # Should be 2-letter codes (e.g., 'CA', 'WI')
                locationmode='USA-states',
                z=full_data['count'],
                zmin=0,
                text=full_data['state'] + ': ' + full_data['count'].astype(str) + ' runs',
                colorscale='bugn',
                hoverinfo='text+z',
                showscale=False
            )

            
            fig.add_trace(choropleth_trace, row=3, col=2)

            fig.update_geos(
            scope='usa',
            showlakes=True,
            lakecolor='lightgray',
            bgcolor='rgba(0,0,0,0)',
            )

            fig.update_layout(
                plot_bgcolor='white',
            )

                    
        # Layout improvements
        fig.update_layout(
            title_text=f"FitWrapped {self.year} Dashboard",
            template='plotly_white',
            height=900, width=1250,
            margin=dict(l=70, r=70, t=100, b=70),  # add more room around the edges
            title_font=dict(size=24), font=dict(size=14),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        fig.update_xaxes(tickangle=-45)
        fig.write_html("fitness_dashboard.html")

if __name__ == "__main__":
    wrapped = FitWrapped()
    wrapped.generate_wrapped()