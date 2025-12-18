import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import matplotlib.font_manager as fm
import platform
import csv

sns.set_style("whitegrid")
plt.switch_backend('Agg')

def set_chinese_font():
    system_fonts = set(f.name for f in fm.fontManager.ttflist)
    font_candidates = [
        'Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'PingFang TC', 
        'Heiti TC', 'Noto Sans TC', 'Noto Sans CJK TC', 'Arial Unicode MS'
    ]
    selected_font = None
    for font in font_candidates:
        if font in system_fonts:
            selected_font = font
            break
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"已設定中文字型: {selected_font}")
    else:
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

def read_csv_auto_encoding(file_path):
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp950', 'big5', 'gb18030']
    separators = [',', '\t'] 
    
    quoting_modes = [csv.QUOTE_MINIMAL, csv.QUOTE_NONE]

    for enc in encodings_to_try:
        for sep in separators:
            for quote_mode in quoting_modes:
                try:
                    df = pd.read_csv(
                        file_path, 
                        encoding=enc, 
                        sep=sep, 
                        quoting=quote_mode,
                        on_bad_lines='skip' 
                    )
                    
                    df.columns = df.columns.astype(str).str.replace('"', '').str.replace("'", '').str.strip()
                    
                    if quote_mode == csv.QUOTE_NONE:
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", '').str.strip()

                    if len(df.columns) <= 1 and df.shape[0] > 0:
                        first_row_str = str(df.iloc[0,0])
                        if ('\t' in first_row_str or ',' in first_row_str) and len(first_row_str) > 10:
                            continue 
                    
                    possible_id_cols = ['user_sn', 'UserSN', 'sn', 'id', 'user_id', '學號']
                    if any(col.lower() in [c.lower() for c in possible_id_cols] for col in df.columns):
                        print(f"✅ 成功讀取 (編碼: {enc}, Sep: {repr(sep)}, Quote: {quote_mode}): {os.path.basename(file_path)}")
                        return df, None
                    if len(df.columns) > 1:
                         print(f"✅ 勉強讀取 (編碼: {enc}, Sep: {repr(sep)}): {os.path.basename(file_path)}")
                         return df, None

                except Exception:
                    continue

    return None, f"無法識別檔案格式 (已嘗試多種編碼與強制切割模式)，請確認檔案是否為有效文字檔。"

def load_target_data(file_log, file_user):
    if file_log is None or file_user is None:
        return None, None, "⚠️ 檔案未上傳完整"

    df_log, err_log = read_csv_auto_encoding(file_log.name)
    if err_log: return None, None, f"行為日誌讀取失敗: {err_log}"
        
    df_user, err_user = read_csv_auto_encoding(file_user.name)
    if err_user: return None, None, f"成績單讀取失敗: {err_user}"

    df_user.columns = [c.strip() for c in df_user.columns]
    df_log.columns = [c.strip() for c in df_log.columns]

    if 'user_sn' not in df_user.columns:
        return None, None, f"❌ 成績單缺少 'user_sn' 欄位。\n偵測到的欄位: {list(df_user.columns)}"
    if 'user_sn' not in df_log.columns:
        if 'user_id' in df_log.columns:
            df_log.rename(columns={'user_id': 'user_sn'}, inplace=True)
        else:
            return None, None, f"❌ 行為日誌缺少 'user_sn' 欄位。\n偵測到的欄位: {list(df_log.columns)}"

    for df in [df_log, df_user]:
        for col in df.columns:
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace('"', '').str.strip()

        df['user_sn'] = df['user_sn'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    if 'review_sn' in df_log.columns:
        df_log['review_sn'] = df_log['review_sn'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    df_user = df_user.drop_duplicates(subset=['user_sn'])
    
    return df_log, df_user, None

def load_baseline_features(file_baseline):
    if file_baseline is None: return None, "無基準檔"
    
    df, err = read_csv_auto_encoding(file_baseline.name)
    if err: return None, f"基準檔讀取錯誤: {err}"
    
    df.columns = df.columns.astype(str).str.replace('"', '').str.strip()
    
    required = ['total_actions', 'unique_videos']
    if not all(col in df.columns for col in required):
        return None, f"基準檔缺少欄位: {required}"
    
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'actions_per_video' not in df.columns:
        df['actions_per_video'] = df.apply(lambda x: x['total_actions'] / x['unique_videos'] if x['unique_videos'] > 0 else 0, axis=1)
        
    return df, None

def process_and_analyze(t_file_log, t_file_user, b_file_csv, manual_score_threshold, use_manual_threshold):
    t_log, t_user, err = load_target_data(t_file_log, t_file_user)
    if err: return err, None, None, None

    feat_actions = t_log.groupby('user_sn').size().reset_index(name='total_actions')
    feat_videos = t_log.groupby('user_sn')['review_sn'].nunique().reset_index(name='unique_videos')
    
    t_features = pd.merge(feat_actions, feat_videos, on='user_sn', how='left').fillna(0)
    t_features['actions_per_video'] = t_features.apply(
        lambda x: x['total_actions'] / x['unique_videos'] if x['unique_videos'] > 0 else 0, axis=1
    )
    t_features['dataset_type'] = 'target'

    combined_features = t_features.copy()
    status_msg = "分析模式：【單獨分析】(無基準)"
    
    if b_file_csv is not None:
        b_features, b_err = load_baseline_features(b_file_csv)
        if not b_err:
            b_features_clean = b_features[['total_actions', 'unique_videos', 'actions_per_video']].copy()
            b_features_clean['dataset_type'] = 'baseline'
            b_features_clean['user_sn'] = 'baseline_' + b_features_clean.index.astype(str)
            combined_features = pd.concat([t_features, b_features_clean], ignore_index=True)
            status_msg = f"分析模式：【菁英常模參照】(Baseline: {len(b_features)}筆)"
        else:
            return f"⚠️ {b_err}", None, None, None

    combined_features['PR_Intensity'] = combined_features['total_actions'].rank(pct=True)
    combined_features['PR_Coverage'] = combined_features['unique_videos'].rank(pct=True)
    combined_features['PR_Density'] = combined_features['actions_per_video'].rank(pct=True)
    combined_features['LPI_Score'] = (
        combined_features['PR_Intensity']*40 + 
        combined_features['PR_Coverage']*30 + 
        combined_features['PR_Density']*30
    ).round(1)

    target_lpi = combined_features[combined_features['dataset_type'] == 'target'].copy()
    
    score_cols = ['chinese_score', 'math_score', 'english_score']
    for col in score_cols:
        if col in t_user.columns:
            t_user[col] = pd.to_numeric(
                t_user[col].astype(str).str.replace('"', '').replace({'NULL': np.nan, 'null': np.nan}), 
                errors='coerce'
            ).fillna(0)
    
    missing_cols = [col for col in score_cols if col not in t_user.columns]
    if missing_cols:
        return f"❌ 成績單缺少欄位: {missing_cols}", None, None, None

    df_final = pd.merge(target_lpi, t_user[['user_sn'] + score_cols], on='user_sn', how='inner')
    df_final['Avg_Score'] = df_final[score_cols].mean(axis=1)
    
    if len(df_final) == 0:
        return "❌ 錯誤：ID 匹配後無資料，請確認 user_sn 是否一致。", None, None, None

    # 象限劃分
    LPI_MID = combined_features['LPI_Score'].median()
    if use_manual_threshold:
        SCORE_MID = float(manual_score_threshold)
        score_source_text = f"自訂 ({SCORE_MID})"
    else:
        SCORE_MID = df_final['Avg_Score'].median()
        score_source_text = f"中位數 ({SCORE_MID:.1f})"

    def get_quadrant(row):
        lpi = row['LPI_Score']
        score = row['Avg_Score']
        if lpi >= LPI_MID and score >= SCORE_MID: return 'Q1: 自主優勢區 (雙高)'
        elif lpi < LPI_MID and score >= SCORE_MID: return 'Q2: 潛力開發區 (低投高產)'
        elif lpi < LPI_MID and score < SCORE_MID: return 'Q3: 基礎扶助區 (雙低)'
        else: return 'Q4: 策略引導區 (高投低產)'

    df_final['Quadrant'] = df_final.apply(get_quadrant, axis=1)
    
    # 輸出 CSV
    out_csv = os.path.join(tempfile.gettempdir(), "LPI_Report_Analysis.csv")
    df_final.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 繪圖
    palette_map = {'Q1: 自主優勢區 (雙高)': '#2E8B57', 'Q2: 潛力開發區 (低投高產)': '#4682B4', 
                   'Q3: 基礎扶助區 (雙低)': "#898980", 'Q4: 策略引導區 (高投低產)': '#D9534F'}
    
    fig_scatter = plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_final, x='LPI_Score', y='Avg_Score', hue='Quadrant', style='Quadrant', 
                    s=120, alpha=0.8, palette=palette_map)
    plt.axvline(LPI_MID, color='grey', linestyle='--', alpha=0.5)
    plt.axhline(SCORE_MID, color='grey', linestyle='--', alpha=0.5)
    plt.title(f'LPI 診斷矩陣 (LPI門檻: {LPI_MID:.1f}, 成績: {score_source_text})', fontsize=16, fontweight='bold')
    plt.tight_layout()

    fig_pie = plt.figure(figsize=(8, 8))
    counts = df_final['Quadrant'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[palette_map.get(x, '#333') for x in counts.index])
    plt.title('分佈比例', fontsize=16)
    plt.tight_layout()

    q4_count = counts.get('Q4: 策略引導區 (高投低產)', 0)
    summary = (f"✅ {status_msg}\n📊 門檻: LPI {LPI_MID:.1f} / 成績 {SCORE_MID:.1f}\n"
               f"👥 分析人數: {len(df_final)}\n⚠️ Q4 關注: {q4_count} 人")
    
    return summary, fig_scatter, fig_pie, out_csv

# --- Gradio ---
css = "footer {display: none !important;}"
with gr.Blocks(title="LPI 學習診斷系統", css=css) as app:
    gr.Markdown("# 🎓 LPI 學習潛能診斷系統 (Pro+)")
    with gr.Row(variant="panel"):
        with gr.Column():
            t_file1 = gr.File(label="行為日誌 (review.csv)")
            t_file2 = gr.File(label="成績單 (user_data.csv)")
        with gr.Column():
            b_file = gr.File(label="基準檔 (elite_baseline.csv)")
            use_manual = gr.Checkbox(label="自訂成績門檻")
            score_threshold = gr.Number(label="分數線", value=60)
            btn = gr.Button("🚀 執行診斷", variant="primary")
            
    with gr.Row():
        out_text = gr.Textbox(label="摘要", lines=5)
        out_file = gr.File(label="下載報表")
    with gr.Row():
        plot_scatter = gr.Plot()
        plot_pie = gr.Plot()

    btn.click(process_and_analyze, inputs=[t_file1, t_file2, b_file, score_threshold, use_manual], 
              outputs=[out_text, plot_scatter, plot_pie, out_file])

if __name__ == "__main__":
    app.launch()