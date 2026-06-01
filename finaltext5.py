import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 🖐️ 第 5 次測試：三段式精準打擊 (終極修正版)
custom_tag = ""
# ==========================================

departments = ['ENT', 'GS', 'GU', 'OPH', 'ORTH']
features_col = [
    '年齡(G)', '性別(H)', '身份(I)', '分類(J)', '麻醉(K)', '手術名稱(L)', 
    '主治醫師(AF)', '醫師年資（月）(AG)', '手術數量(AQ)', '分類(AZ)', '醫師人數(BG)'
]
target_col = '手術時間（分）(BQ)' 
performance_report = []

for dept in departments:
    print(f"\n{'='*10} 正在處理科別: {dept} {'='*10}")
    
    train_filename = f"訓練集-{dept}_Training.csv"
    test_filename = f"測試集-{dept}_Testing.csv"
    
    try:
        try:
            train_df = pd.read_csv(train_filename, encoding='utf-8')
            test_df = pd.read_csv(test_filename, encoding='utf-8')
        except UnicodeDecodeError:
            train_df = pd.read_csv(train_filename, encoding='big5')
            test_df = pd.read_csv(test_filename, encoding='big5')
    except FileNotFoundError:
        print(f"❌ 找不到檔案，跳過: {dept}")
        continue

    # 1. 欄位校正
    rename_map = {'手術數量(BP)': '手術數量(AQ)', '分類(AY)': '分類(AZ)', '醫師人數(BO)': '醫師人數(BG)'}
    train_df.rename(columns=rename_map, inplace=True)
    test_df.rename(columns=rename_map, inplace=True)

    # ========================================================
    # 🔥【第 5 次核心修改】：三段式策略
    # ========================================================
    original_count = len(train_df)
    
    # 策略 A: GS, ORTH -> 維持第 2/4 次的最佳設定 (0.95)
    if dept in ['GS', 'ORTH']:
        limit = train_df[target_col].quantile(0.95) 
        train_df = train_df[train_df[target_col] < limit]
        print(f"🧹 {dept} (去離群 0.95): 移除 >95% ({limit:.1f}分) 資料 -> 鎖定歷史最佳")
        
    # 策略 B: ENT -> 回歸第 1 次的最佳設定 (0.97)
    # 🔥 這是救回 ENT 42分慘劇的關鍵！
    elif dept == 'ENT':
        limit = train_df[target_col].quantile(0.97) 
        train_df = train_df[train_df[target_col] < limit]
        print(f"🚑 {dept} (微創去離群 0.97): 移除 >97% ({limit:.1f}分) 資料 -> 目標回歸 34.93")
        
    # 策略 C: GU, OPH -> 維持第 1/4 次的最佳設定 (全保留)
    # GU 剛拿到 39.03，絕對不能動它
    else:
        print(f"🛡️ {dept} (全資料): 保留完整極端值 -> 鎖定歷史最佳")
    # ========================================================

    # 資料清理
    train_df['is_train'] = 1; test_df['is_train'] = 0; test_df[target_col] = 0 
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    current_features = [c for c in features_col if c in full_df.columns]
    num_cols = full_df[current_features].select_dtypes(include=[np.number]).columns
    cat_cols = full_df[current_features].select_dtypes(exclude=[np.number]).columns
    
    full_df[num_cols] = full_df[num_cols].fillna(0)
    full_df[cat_cols] = full_df[cat_cols].fillna('Unknown')
    
    le = LabelEncoder()
    for col in cat_cols:
        full_df[col] = full_df[col].astype(str)
        full_df[col] = le.fit_transform(full_df[col])
        
    train_final = full_df[full_df['is_train'] == 1].copy()
    test_final = full_df[full_df['is_train'] == 0].copy()
    
    X = train_final[current_features]
    y = train_final[target_col]
    X_real_test = test_final[current_features]
    
    # 模型參數：維持最穩定的設定 (深度維持 Run 4 的設定)
    depth_setting = 30 if dept in ['ENT', 'GS', 'GU'] else 20
    
    model = RandomForestRegressor(n_estimators=500, max_depth=depth_setting, min_samples_leaf=5, random_state=42, n_jobs=-1)

    # 驗證
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    performance_report.append({'Dept': dept, 'MAE': mae})
    
    # 輸出
    model.fit(X, y)
    final_preds = model.predict(X_real_test)
    pd.DataFrame({'Predicted_Time': final_preds}).to_csv(f'第五次預測結果_{dept}{custom_tag}.csv', index=False)
    print(f"✅ {dept} 完成，預測 MAE: {mae:.2f}")

print(pd.DataFrame(performance_report))
