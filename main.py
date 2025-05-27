import streamlit as st
import pandas as pd
from sklearn import KMeans
import plotly.express as px

st.set_page_config(page_title="배달 위치 군집 시각화", layout="wide")

st.title("📦 배달 위치 군집 시각화 웹앱")

st.markdown("""
이 앱은 배달 위치 좌표 데이터를 K-Means로 군집화하여 지도에 시각화합니다.  
좌표 데이터를 담은 CSV 파일을 업로드하면 자동으로 군집화 결과를 확인할 수 있습니다.
""")

# 파일 업로드
uploaded_file = st.file_uploader("📤 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 유효성 검사
    required_columns = {'Latitude', 'Longitude'}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ CSV 파일에 다음 컬럼이 필요합니다: {required_columns}")
    else:
        st.success("✅ 파일 업로드 완료! 군집 수를 선택해 주세요.")

        # 군집 수 슬라이더
        k = st.slider("군집 수 선택 (K)", min_value=2, max_value=10, value=3)

        # K-Means 군집화
        coords = df[['Latitude', 'Longitude']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(coords)

        # 지도 시각화
        fig = px.scatter_mapbox(
            df,
            lat='Latitude',
            lon='Longitude',
            color=df['Cluster'].astype(str),
            zoom=11,
            mapbox_style="carto-positron",
            title=f"K={k} 클러스터 군집 시각화",
            width=1000,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # 클러스터별 데이터 개수 출력
        st.subheader("📊 클러스터별 데이터 수")
        st.dataframe(df['Cluster'].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Count"))

else:
    st.info("좌표 데이터를 포함한 CSV 파일을 업로드해 주세요.")
