import streamlit as st
import plotly.graph_objects as go

import yt_utils
import analyze_title as at
import analyze_content as ac

st.set_page_config(page_title="WorthIt", page_icon="🕵️", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = {}
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

with st.sidebar:
    st.title("WorthIt")
    st.caption("Clickbait detector.")
    st.divider()

    st.markdown("### Search History")
    for title_hist, url_hist in list(st.session_state.history.items()):
        if st.button(f"{title_hist[:25]}...", key=url_hist, use_container_width=True):
            st.session_state.current_url = url_hist
            st.rerun()

    if st.session_state.history:
        st.write("")
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = {}
            st.session_state.current_url = ""
            st.rerun()

st.title("🕵️ WorthIt")
st.write("")

url_input = st.text_input("Analyze YT video URL:",
                          value=st.session_state.current_url,
                          placeholder="Paste YouTube link and press Enter...",
                          label_visibility="collapsed")

if url_input != st.session_state.current_url:
    st.session_state.current_url = url_input
    st.rerun()

if not st.session_state.current_url:
    st.stop()

url = st.session_state.current_url.strip()

with st.spinner("Analyzing video..."):
    video_id = yt_utils.get_video_id(url)
    title = yt_utils.get_title(video_id)
    st.session_state.history[title] = url

    title_results = at.analyze_title(title)
    title_score = title_results["score"]

    content_results = ac.analyze_content(title, video_id, verbose=False)

    if content_results is None:
        st.error("Could not analyze this video. It might not have English subtitles/transcription available.")
        if st.button("Try another video", use_container_width=True):
            st.session_state.current_url = ""
            st.rerun()
        st.stop()

    chunks = content_results["chunks"]
    scores = content_results["scores"]
    best_i = content_results["best_index"]
    best_chunk = content_results["best_chunk"]

    peak_match_pct = content_results["peak_match_pct"]
    avg_match_pct = content_results["avg_match_pct"]
    topic_density_pct = content_results["topic_density_pct"]
    signal_chaos_pct = content_results["signal_chaos_pct"]
    times_min = [round(c["start"] / 60, 2) for c in chunks]
    timestamp_formatted = ac.format_time(best_chunk["start"])
    
st.write("---")
st.markdown(f"### Title: **{title}**")
st.write("---")

st.markdown("#### Analysis Metrics")
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Headline Clickbait Score", f"{int(title_score * 100)}%")
col_m2.metric("Peak Content Match", f"{peak_match_pct}%")
col_m3.metric("Average Match", f"{avg_match_pct}%")
col_m4.metric("Topic Focus", f"{topic_density_pct}%")
col_m5.metric("Signal Chaos", f"{signal_chaos_pct}%")

st.write("")

col_chart, col_time = st.columns([2, 1], gap="large")

with col_chart:
    with st.container(border=True):
        st.markdown("**Timeline Relevance (X: Minutes | Y: Match Strength)**")
        fig_timeline = go.Figure()

        fig_timeline.add_trace(go.Scatter(
            x=times_min, y=scores, mode='lines',
            line=dict(width=3, color='royalblue', shape='spline'),
            name="Relevance", hovertemplate="Minute: %{x}<br>Match: %{y:.2f}<extra></extra>"
        ))

        fig_timeline.add_trace(go.Scatter(
            x=[times_min[best_i]], y=[scores[best_i]], mode='markers',
            marker=dict(color='white', size=10, line=dict(color='crimson', width=2)),
            name="Peak Match", hoverinfo='skip'
        ))

        fig_timeline.update_layout(
            height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='lightgray'),
                       title="Video Timeline (Minutes)"),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 1], tickfont=dict(color='lightgray'),
                tickvals=[0.0, ac.WEAK_THRESHOLD, ac.STRONG_THRESHOLD, 1.0],
                ticktext=["0.0", "Weak Match", "Strong Match", "1.0"]
            )
        )
        st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

        with st.container(border=True):
            st.markdown("### Is it worth watching?")

            watch_verdict = ac.compute_watch_verdict(title_score, content_results)
            st.markdown(watch_verdict["headline"])
            st.markdown(watch_verdict["message"])
    if st.button("Analyze another video", use_container_width=True, type="primary"):
        st.session_state.current_url = ""
        st.rerun()

with col_time:
    with st.container(border=True):
        st.markdown("**Highest Relevance Point**")
        if scores[best_i] >= ac.WEAK_THRESHOLD:
            st.header(f"⏱️ {timestamp_formatted}")
            st.caption("The most important point of the video starts here. Context:")
            st.info(f"\"{best_chunk['text']}\"")
        else:
            st.header("None")
            st.caption("No strictly relevant moment found.")
