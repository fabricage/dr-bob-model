"""Streamlit dashboard for the local NFL spread model.

Three views:
  1. This week's board — model vs market, edge, overrides
  2. Team page — blended ratings, raw vs compensated, weekly trend
  3. Research — stability tables, weight curve, backtest

This app is research / decision-support only. It does not place bets.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import polars as pl
import streamlit as st

import config
from model.spread import week_board
from model.weights import blended_rating, load_prior_r

st.set_page_config(page_title="NFL spread model", layout="wide")

def _line_chart(df: pl.DataFrame, x: str, y: str, y_title: str, y_domain: list[float] | None = None) -> None:
    """Altair line so axes stay in the units we meant (not a mystery index)."""
    pdf = df.select([x, y]).to_pandas()
    y_enc = alt.Y(f"{y}:Q", title=y_title)
    if y_domain is not None:
        y_enc = alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=y_domain))
    chart = (
        alt.Chart(pdf)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ")),
            y=y_enc,
            tooltip=[x, y],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _read(path, what: str) -> pl.DataFrame | None:
    if not path.exists():
        st.warning(f"Missing {what} at `{path}`. Run the pipeline (see README) first.")
        return None
    return pl.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_board_source() -> pl.DataFrame | None:
    return _read(config.backtest_games_path(), "scored games")


@st.cache_data(show_spinner=False)
def load_games() -> pl.DataFrame | None:
    return _read(config.team_game_stats_path(), "compensated team-game stats")


@st.cache_data(show_spinner=False)
def load_raw_games() -> pl.DataFrame | None:
    return _read(config.team_game_stats_raw_path(), "raw team-game stats")


@st.cache_data(show_spinner=False)
def load_effects() -> pl.DataFrame | None:
    return _read(config.team_effects_path(), "team effects")


@st.cache_data(show_spinner=False)
def load_stability() -> pl.DataFrame | None:
    return _read(config.stability_oddeven_path(), "odd/even stability")


@st.cache_data(show_spinner=False)
def load_curve() -> pl.DataFrame | None:
    return _read(config.weight_curve_path(), "first-N weight curve")


@st.cache_data(show_spinner=False)
def load_backtest() -> pl.DataFrame | None:
    return _read(config.backtest_path(), "backtest report")


def _latest_season_week(scored: pl.DataFrame) -> tuple[int, int]:
    last = scored.sort(["season", "week"]).tail(1)
    return int(last["season"][0]), int(last["week"][0])


def view_board() -> None:
    st.title("This week's board")
    st.caption("Model home spread vs market (schedule closing line). Sorted by absolute edge.")
    scored = load_board_source()
    if scored is None:
        return
    seasons = sorted(scored.get_column("season").unique().to_list())
    default_season, default_week = _latest_season_week(scored)
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", seasons, index=seasons.index(default_season))
    weeks = sorted(
        scored.filter(pl.col("season") == season).get_column("week").unique().to_list()
    )
    week = c2.selectbox(
        "Week",
        weeks,
        index=weeks.index(default_week) if default_week in weeks else len(weeks) - 1,
    )
    board = week_board(int(season), int(week), scored=scored)
    if board.height == 0:
        st.info("No games for that week.")
        return
    show_cols = [
        c
        for c in (
            "away_team",
            "home_team",
            "market_home_spread",
            "model_home_spread",
            "adjusted_home_spread",
            "home_spread_edge",
            "home_spread_edge_adjusted",
            "predicted_margin",
            "actual_margin",
            "n_overrides",
            "override_reasons",
        )
        if c in board.columns
    ]
    pretty = board.select(show_cols).with_columns(pl.col(pl.Float64).round(2))
    st.dataframe(pretty.to_pandas(), use_container_width=True, hide_index=True)
    flagged = board.filter(pl.col("n_overrides") > 0) if "n_overrides" in board.columns else board.head(0)
    if flagged.height:
        st.subheader("Active overrides this week")
        st.dataframe(flagged.select(show_cols).to_pandas(), use_container_width=True, hide_index=True)
    st.caption(
        "Sign convention: market/model home spread is the betting number "
        "(negative = home favored). Edge = market − model. "
        "Positive edge means the model likes the home team vs the market."
    )


def view_team() -> None:
    st.title("Team page")
    games = load_games()
    raw = load_raw_games()
    effects = load_effects()
    if games is None:
        return
    teams = sorted(games.get_column("team").unique().to_list())
    seasons = sorted(games.get_column("season").unique().to_list())
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", teams, index=teams.index("KC") if "KC" in teams else 0)
    season = c2.selectbox("Season", seasons, index=len(seasons) - 1)

    season_games = games.filter((pl.col("season") == season) & (pl.col("team") == team)).sort("week")
    st.subheader(f"{team} {season} — weekly compensated EPA")
    if season_games.height:
        trend = season_games.select(
            "week",
            pl.col("off_epa_per_play_comp").round(3).alias("off_epa_comp"),
            pl.col("def_epa_per_play_comp").round(3).alias("def_epa_comp"),
            pl.col("off_epa_per_play").round(3).alias("off_epa_raw"),
            pl.col("def_epa_per_play").round(3).alias("def_epa_raw"),
        )
        long = trend.unpivot(
            index="week",
            on=["off_epa_comp", "def_epa_comp", "off_epa_raw", "def_epa_raw"],
            variable_name="series",
            value_name="epa",
        )
        chart = (
            alt.Chart(long.to_pandas())
            .mark_line(point=True)
            .encode(
                x=alt.X("week:Q", title="week"),
                y=alt.Y("epa:Q", title="EPA/play"),
                color="series:N",
                tooltip=["week", "series", "epa"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(trend.to_pandas(), use_container_width=True, hide_index=True)

    st.subheader("Raw vs compensated season averages")
    if raw is not None:
        raw_s = raw.filter((pl.col("season") == season) & (pl.col("team") == team))
        rows = []
        for stat in config.EFFICIENCY_STATS:
            off_raw = raw_s.get_column(f"off_{stat}").mean()
            off_comp = season_games.get_column(f"off_{stat}_comp").mean() if season_games.height else None
            def_raw = raw_s.get_column(f"def_{stat}").mean()
            def_comp = season_games.get_column(f"def_{stat}_comp").mean() if season_games.height else None
            rows.append(
                {
                    "stat": stat,
                    "off_raw": off_raw,
                    "off_comp": off_comp,
                    "def_raw": def_raw,
                    "def_comp": def_comp,
                }
            )
        st.dataframe(
            pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4)).to_pandas(),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Blended rating as of a week")
    weeks = list(range(1, config.last_usable_week(int(season)) + 1))
    as_of = st.slider("As-of week (uses games strictly before this week)", min_value=1, max_value=max(weeks), value=min(8, max(weeks)))
    try:
        off = blended_rating(team, "off_epa_per_play", int(season), int(as_of))
        deff = blended_rating(team, "def_epa_per_play", int(season), int(as_of))
        st.write(
            {
                "games_played": off["games_played"],
                "weight_on_current_season": round(off["weight_current"], 3),
                "off_epa_prior": round(off["prior"], 4),
                "off_epa_current": round(off["current_mean"], 4),
                "off_epa_blended": round(off["blended"], 4),
                "def_epa_blended": round(deff["blended"], 4),
                "net_epa_blended": round(off["blended"] - deff["blended"], 4),
                "max_week_used": off["max_week_used"],
            }
        )
    except FileNotFoundError as exc:
        st.info(str(exc))

    if effects is not None:
        eff = effects.filter((pl.col("season") == season) & (pl.col("team") == team))
        st.subheader("Season-level ridge effects")
        st.dataframe(
            eff.select("stat", "offense_effect", "defense_effect", "hfa")
            .with_columns(pl.col(pl.Float64).round(4))
            .to_pandas(),
            use_container_width=True,
            hide_index=True,
        )


def view_research() -> None:
    st.title("Research")
    st.caption("Stability studies and the walk-forward backtest — the evidence under the ratings.")

    st.subheader("Odd/even split-half reliability")
    stab = load_stability()
    if stab is not None:
        st.dataframe(
            stab.with_columns(pl.col(pl.Float64).round(3)).to_pandas(),
            use_container_width=True,
            hide_index=True,
        )
        bar = (
            alt.Chart(stab.to_pandas())
            .mark_bar()
            .encode(
                x=alt.X("r_spearman_brown:Q", title="Spearman–Brown reliability", scale=alt.Scale(domain=[-0.2, 1.0])),
                y=alt.Y("stat:N", sort="-x", title="stat"),
                tooltip=["stat", "r_spearman_brown", "r_split_half"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)
        st.caption("EPA/play and success rate should beat fumbles_lost and turnover_margin.")

    st.subheader("First-N → rest-of-season (weight curve)")
    curve = load_curve()
    if curve is not None:
        wide = (
            curve.with_columns(pl.col("r_first_n").round(3))
            .pivot(values="r_first_n", index="stat", on="n_games", sort_columns=True)
        )
        st.dataframe(wide.to_pandas(), use_container_width=True, hide_index=True)
        epa = curve.filter(pl.col("stat") == "off_epa_per_play").sort("n_games")
        if epa.height:
            _line_chart(epa, "n_games", "r_first_n", "r(first N, rest of season)", y_domain=[-0.1, 1.0])
            try:
                from model.weights import weight_curve_table

                weights = weight_curve_table(curve, load_prior_r(), "off_epa_per_play")
                st.subheader("Blend weight vs games played (off_epa_per_play)")
                _line_chart(
                    weights,
                    "games_played",
                    "weight_current",
                    "weight on current season",
                    y_domain=[0.0, 1.0],
                )
            except FileNotFoundError:
                pass

    st.subheader("Walk-forward backtest")
    report = load_backtest()
    if report is not None:
        st.dataframe(
            report.with_columns(pl.col(pl.Float64).round(3)).to_pandas(),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("ATS figures are research diagnostics, not a betting record to chase.")


def view_build() -> None:
    """Download data and rebuild the model from the browser, not the terminal."""
    st.title("Build data")
    st.caption(
        "This page downloads NFL data and rebuilds ratings. Leave the terminal "
        "window that launched the app open — that window is only the server."
    )
    status = config.data_status()
    ready = config.board_ready()
    if ready:
        st.success("Model files are on disk. The other pages should show real games.")
    else:
        st.info(
            "First time here: click **First-time build**. It needs internet and "
            "usually takes several minutes (2015–2025 play-by-play)."
        )
    rows = [{"file": name, "ready": "yes" if ok else "no"} for name, ok in status.items()]
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    first = c1.button("First-time build (download + full model)", type="primary")
    weekly = c2.button("Weekly refresh (current season)")

    if first or weekly:
        log_box = st.empty()
        argv = ["--full"] if first else []
        buf: list[str] = []

        class _Tee:
            def write(self, chunk: str) -> int:
                if chunk:
                    buf.append(chunk)
                    log_box.code("".join(buf)[-12000:])
                return len(chunk)

            def flush(self) -> None:
                return None

        with st.spinner("Working… this page will update as steps finish."):
            old = sys.stdout
            sys.stdout = _Tee()  # type: ignore[assignment]
            try:
                from run_week import main as run_week_main

                code = run_week_main(argv)
            finally:
                sys.stdout = old
        if code == 0:
            st.cache_data.clear()
            st.success("Done. Open **This week's board** in the sidebar.")
        else:
            st.error(f"Build exited with code {code}. Scroll the log above.")


def main() -> None:
    st.sidebar.title("NFL spread model")
    st.sidebar.info(
        "This is a web page at http://localhost:8501. "
        "The terminal is only the server — leave it open. "
        "No accounts, no bet placement."
    )
    views = ["This week's board", "Team page", "Research", "Build data"]
    if not config.board_ready():
        views = ["Build data", "This week's board", "Team page", "Research"]
    page = st.sidebar.radio("View", views)
    if page == "This week's board":
        view_board()
    elif page == "Team page":
        view_team()
    elif page == "Research":
        view_research()
    else:
        view_build()


if __name__ == "__main__":
    main()
