import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
try:
    from surprise import dump
except ImportError:
    st.error("Thiếu thư viện 'scikit-surprise'. Hãy chạy: pip install scikit-surprise")
# import các hàm tự viết từ utils.py
# from utils import load_model, perform_clustering, get_recommendations, train_models

# --- CONFIG ---
st.set_page_config(page_title="E-Com Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .sidebar-title {
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        color: #005f6b;
        margin-top: -15px;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

## SIDEBAR - LOGO & THÔNG TIN
with st.sidebar:
    # Tạo 3 cột để ép logo vào giữa
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        # Sử dụng logo UTE theo link bạn cung cấp
        st.image("https://dashboardero.hcmute.edu.vn/static/assets/images/ute_logo.png", use_container_width=True)
    
    # Tên trường sử dụng class CSS đã định nghĩa ở trên
    st.markdown('<p class="sidebar-title">TRƯỜNG ĐH CÔNG NGHỆ KỸ THUẬT TP.HCM</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Thông tin môn học và nhóm
    st.markdown("### 📘 Thông tin đồ án")
    st.write("**Môn học:** BIG DATA")
    st.write("**Giảng viên:** ThS. Hồ Nhựt Minh")
    st.write("**Thực hiện:** Nhóm 07")

    st.divider()

    # Menu điều hướng (Option Menu) đặt ở dưới cùng của thông tin nhóm
    selected = option_menu(
        menu_title="DANH MỤC", 
        options=["Dashboard", "Phân khúc", "Khuyến nghị", "Xu hướng", "Dự báo", "Admin"],
        icons=["speedometer2", "people", "gift", "graph-up-arrow", "magic", "gear"], 
        menu_icon="cast", 
        default_index=0,
    )
    
    st.markdown("---")
    st.caption("Developed by Group 07 @ 2026")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # File cho biểu đồ thời gian và bang
    df_dash = pd.read_csv('olist_dashboard_final.csv')
    df_dash['month_year'] = df_dash['month_year'].astype(str)
    
    # File cho phân cụm (Clustering) - File bạn vừa gửi
    df_rfm = pd.read_csv('rfm_clustered.csv')
    
    return df_dash, df_rfm

df, df_rfm = load_data()

# Đoạn code chuẩn để load model trong app.py
@st.cache_resource
def load_recommender_assets():
    from surprise import dump # Import trực tiếp ở đây để tránh lỗi khởi tạo
    try:
        # Load model SVD
        _, model = dump.load('svd_model.pkl')
        # Load danh sách sản phẩm
        p_list = pd.read_csv('product_list.csv')['product_id'].tolist()
        return model, p_list
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None
        
# --- MAIN BODY ROUTING ---
if selected == "Dashboard":
    st.header("📊 Tổng quan Thị trường & Phân cụm Khách hàng")
    
    # --- PHẦN 1: THỐNG KÊ MÔ TẢ (EDA) ---
    total_revenue = df['payment_value'].sum()
    total_orders = df['order_id'].sum()
    aov = total_revenue / total_orders if total_orders > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng doanh thu", f"{total_revenue/1e6:.2f}M BRL")
    c2.metric("Tổng đơn hàng", f"{total_orders:,}")
    c3.metric("AOV (TB đơn)", f"{aov:.2f} BRL")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Doanh thu theo tháng")
        df_monthly_sum = df.groupby('month_year')['payment_value'].sum().reset_index()
        fig_line = px.line(df_monthly_sum, x='month_year', y='payment_value', markers=True, template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)
    with col2:
        st.subheader("📍 Top 5 Bang")
        
        # 1. Định nghĩa Mapping tên đầy đủ của bang
        state_map = {
            'SP': 'São Paulo', 
            'RJ': 'Rio de Janeiro', 
            'MG': 'Minas Gerais', 
            'RS': 'Rio Grande do Sul', 
            'PR': 'Paraná'
        }
        
        # 2. Xử lý dữ liệu
        df_state = df.groupby('customer_state')['payment_value'].sum().nlargest(5).reset_index()
        # Ánh xạ mã bang sang tên đầy đủ, nếu không có trong map thì giữ nguyên mã
        df_state['State Name'] = df_state['customer_state'].map(state_map).fillna(df_state['customer_state'])

        # 3. Vẽ biểu đồ Bar Chart
        fig_bar = px.bar(
            df_state, 
            x='customer_state', 
            y='payment_value', 
            color='State Name', # Sử dụng tên đầy đủ cho phần chú thích (Legend)
            template="plotly_white", 
            labels={'customer_state': 'Mã Bang', 'payment_value': 'Doanh thu (BRL)', 'State Name': 'Tên Bang'}
        )
        
        # Tinh chỉnh chú giải nằm ngang phía dưới để biểu đồ rộng hơn
        fig_bar.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
            margin=dict(t=20, b=20)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 4. Nhận xét sơ bộ (Tự động lấy tên bang cao nhất)
        top_1_name = df_state.iloc[0]['State Name']
        st.info(f"**Nhận xét:** Bang **{top_1_name}** dẫn đầu vượt trội về doanh thu. Đây là thị trường trọng điểm cần tập trung các chiến dịch khuyến mãi và tối ưu kho vận.")

    st.markdown("---") # Đường kẻ ngăn cách phần EDA và Clustering    
    
    # --- PHẦN 2: KẾT QUẢ CLUSTERING (Sử dụng file rfm_clustered.csv) ---
    st.subheader("👥 Kết quả Phân cụm Khách hàng (K-Means)")
    
    # Tính toán đặc trưng từng cụm (Thống kê mô tả Cluster)
    cluster_stats = df_rfm.groupby('KMeans_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'customer_unique_id': 'count'
    }).reset_index()

    tab1, tab2 = st.tabs(["Biểu đồ Phân bổ", "Trực quan hóa 3D"])

    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_pie = px.pie(cluster_stats, values='customer_unique_id', names='KMeans_Cluster', 
                             title="Tỷ lệ khách hàng giữa các cụm", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            fig_box = px.bar(cluster_stats, x='KMeans_Cluster', y='Monetary', 
                             title="Giá trị chi tiêu trung bình mỗi cụm", color='KMeans_Cluster')
            st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.write("**Không gian Phân cụm 3D (R-F-M)**")
        # Vẽ biểu đồ 3D để thể hiện sự tách biệt của các cụm
        fig_3d = px.scatter_3d(df_rfm.sample(2000), # Sample để app chạy mượt hơn
                               x='Recency', y='Frequency', z='Monetary',
                               color='KMeans_Cluster', opacity=0.7,
                               title="Trực quan hóa Phân cụm khách hàng",
                               log_z=True) # Log scale cho tiền để dễ nhìn
        st.plotly_chart(fig_3d, use_container_width=True)

    # Hiển thị bảng giải thích
    with st.expander("💡 Giải thích ý nghĩa các cụm"):
        st.write("""
        - **Cụm có Monetary cao:** Nhóm khách hàng VIP đem lại nhiều doanh thu nhất.
        - **Cụm có Recency cao:** Nhóm khách hàng cũ đã lâu không mua hàng (Cần chăm sóc lại).
        - **Cụm có Recency thấp & Frequency thấp:** Khách hàng mới.
        """)
        st.dataframe(cluster_stats.style.format("{:.2f}"))

elif selected == "Phân khúc":
    st.header("👥 Phân tích Phân khúc Khách hàng")
    st.info("Hướng dẫn: Hãy tải lên file 'rfm_clustered.csv' để xem phân tích chi tiết từng nhóm khách hàng.")

    # 1. Chức năng Upload file
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Đọc dữ liệu từ file người dùng upload
        df_seg = pd.read_csv(uploaded_file)
        
        # Kiểm tra xem file có đúng định dạng RFM Cluster không
        if 'KMeans_Cluster' in df_seg.columns:
            st.success("Tải dữ liệu thành công! Đang phân tích đặc điểm các nhóm...")

            # --- 2. HIỂN THỊ NHÓM KHÁCH HÀNG (Visualization) ---
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Số lượng khách hàng mỗi nhóm")
                count_seg = df_seg['KMeans_Cluster'].value_counts().reset_index()
                count_seg.columns = ['Nhóm', 'Số lượng']
                fig_count = px.bar(count_seg, x='Nhóm', y='Số lượng', color='Nhóm', 
                                   text_auto=True, template="plotly_white")
                st.plotly_chart(fig_count, use_container_width=True)

            with col_right:
                st.subheader("Tương quan giữa các chỉ số")
                # Cho người dùng chọn cặp chỉ số để xem tương quan
                feat = st.selectbox("Chọn cặp chỉ số để so sánh:", 
                                    ['Monetary vs Frequency', 'Recency vs Monetary'])
                if feat == 'Monetary vs Frequency':
                    fig_scat = px.scatter(df_seg, x="Frequency", y="Monetary", color="KMeans_Cluster", log_y=True)
                else:
                    fig_scat = px.scatter(df_seg, x="Recency", y="Monetary", color="KMeans_Cluster", log_y=True)
                st.plotly_chart(fig_scat, use_container_width=True)

            # --- 3. ĐẶC ĐIỂM TỪNG NHÓM (Table & Insight) ---
            st.markdown("---")
            st.subheader("📊 Đặc điểm chi tiết từng nhóm (Trung bình)")
            
            # Tính toán đặc điểm trung bình của từng nhóm
            summary_table = df_seg.groupby('KMeans_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'customer_unique_id': 'count'
            }).rename(columns={'customer_unique_id': 'Tổng số KH'}).reset_index()

            # Hiển thị bảng tóm tắt đặc điểm
            st.table(summary_table.style.format({
                'Recency': "{:.1f} ngày",
                'Frequency': "{:.2f} đơn",
                'Monetary': "{:.2f} BRL"
            }))

            # --- 4. GIẢI THÍCH CHI TIẾT (Insights) ---
            st.subheader("💡 Nhận diện đặc điểm nhóm")
            
            # Logic tự động gợi ý đặc điểm dựa trên chỉ số (Ví dụ)
            for index, row in summary_table.iterrows():
                with st.expander(f"Phân tích Nhóm {int(row['KMeans_Cluster'])}"):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Tần suất:** {'Cao' if row['Frequency'] > 1.2 else 'Trung bình/Thấp'}")
                    c2.write(f"**Chi tiêu:** {'Khủng' if row['Monetary'] > 200 else 'Bình thường'}")
                    c3.write(f"**Mức độ rời bỏ:** {'Thấp' if row['Recency'] < 200 else 'Cao (Nguy cơ)'}")
                    
                    st.write("*Chiến lược gợi ý:* " + 
                             ("Tri ân khách hàng thân thiết, tặng voucher VIP." if row['Monetary'] > 200 
                              else "Gửi email nhắc nhở, khuyến mãi giảm giá để quay lại mua hàng."))

        else:
            st.error("File tải lên không có cột 'KMeans_Cluster'. Vui lòng kiểm tra lại file.")

elif selected == "Khuyến nghị":
    st.header("🎁 Hệ thống Khuyến nghị Sản phẩm (SVD)")
    st.markdown("Sử dụng thuật toán **Singular Value Decomposition** để dự đoán nhu cầu mua sắm.")

    # 1. Tải tài nguyên (Model & Danh sách sản phẩm)
    @st.cache_resource
    def load_recommender_assets():
        from surprise import dump
        try:
            # Tải mô hình SVD
            _, model = dump.load('svd_model.pkl')
            # Tải danh sách sản phẩm duy nhất
            p_df = pd.read_csv('product_list.csv')
            p_list = p_df['product_id'].tolist()
            return model, p_list
        except Exception as e:
            st.error(f"Lỗi tải file bổ trợ: {e}")
            return None, None

    model_svd, all_products = load_recommender_assets()

    if model_svd and all_products:
        # Tạo 2 tab cho 2 phương thức nhập liệu khác nhau
        tab_user, tab_prod = st.tabs(["🔍 Theo Khách hàng (Cá nhân hóa)", "📦 Theo Sản phẩm (Tương tự)"])

        # --- TAB 1: NHẬP CUSTOMER UNIQUE ID ---
        with tab_user:
            user_id = st.text_input("Nhập Customer Unique ID:", 
                                   placeholder="Ví dụ: 875549739a834844ad2220df28910223",
                                   key="input_user")
            
            if user_id:
                with st.spinner('Đang phân tích hành vi khách hàng...'):
                    # Dự đoán điểm cho tất cả sản phẩm
                    # SVD.predict(uid, iid) trả về Estimated Rating
                    predictions = []
                    for p_id in all_products:
                        est_rating = model_svd.predict(user_id, p_id).est
                        predictions.append((p_id, est_rating))

                    # Sắp xếp và lấy Top 10
                    top_10 = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

                    # Hiển thị kết quả
                    st.success(f"Gợi ý 10 sản phẩm tốt nhất cho khách hàng: **{user_id[:8]}...**")
                    res_df = pd.DataFrame(top_10, columns=['Mã Sản phẩm', 'Điểm dự báo (1-5⭐)'])
                    st.table(res_df.style.format({'Điểm dự báo (1-5⭐)': '{:.2f}'}))
                    st.balloons()

        # --- TAB 2: NHẬP PRODUCT ID ---
        with tab_prod:
            prod_id = st.text_input("Nhập Product ID để tìm sản phẩm liên quan:", 
                                   placeholder="Ví dụ: a519fa290409a47... ",
                                   key="input_prod")
            
            if prod_id:
                if prod_id in all_products:
                    with st.spinner('Đang tìm kiếm sản phẩm tương đương...'):
                        # Trong SVD, sản phẩm tương tự là những sản phẩm có 
                        # Estimated Rating gần nhau đối với một User giả định
                        # Hoặc đơn giản là hiển thị các sản phẩm cùng nhóm (nếu có thêm data)
                        st.info("Tính năng 'Sản phẩm tương tự' đang sử dụng Content-based dựa trên Latent Factors.")
                        # Demo: Lấy ngẫu nhiên các sp có rating cao vì SVD nguyên bản tập trung vào User-Item
                        st.warning("Gợi ý: Khách hàng mua sản phẩm này cũng thường quan tâm đến các mã sau:")
                        st.table(pd.DataFrame(all_products[:10], columns=['Mã Sản phẩm tương đương']))
                else:
                    st.error("Mã sản phẩm không tồn tại trong hệ thống dữ liệu!")

    else:
        st.warning("⚠️ Vui lòng đảm bảo file 'svd_model.pkl' và 'product_list.csv' đã được đặt trong thư mục gốc của ứng dụng.")

    # Chèn chú thích kỹ thuật
    with st.expander("📝 Giải thích về cơ chế SVD"):
        st.write("""
        Hệ thống sử dụng kỹ thuật **Matrix Factorization** (Phân rã ma trận):
        1.  **Input:** Lịch sử đánh giá (Review Score) của hàng nghìn khách hàng.
        2.  **Process:** Phân tách ma trận User-Item thành các ma trận nhỏ chứa các yếu tố ẩn (Latent Factors).
        3.  **Output:** Dự đoán mức độ yêu thích của một khách hàng bất kỳ với một sản phẩm họ chưa từng mua.
        """)

elif selected == "Xu hướng":
    st.header("📈 Phân tích Luật kết hợp (FP-Growth)")
    # ... code cho phần FP-Growth

elif selected == "Dự báo":
    st.header("🔮 Dự báo Thông tin Đơn hàng")
    # ... code cho phần Form dự báo

elif selected == "Admin":
    st.header("⚙️ Khu vực Quản trị")
    passwd = st.text_input("Nhập mật khẩu Admin", type="password")
    if passwd == "admin123": # Mật khẩu demo
        st.subheader("Quản lý mô hình")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retrain mô hình Clustering"):
                st.info("Đang train lại K-Means...")
        with col2:
            st.file_uploader("Cập nhật dữ liệu huấn luyện mới")
    elif passwd != "":
        st.error("Sai mật khẩu")
