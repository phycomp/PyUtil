import streamlit as st
import subprocess
import tempfile
from pathlib import Path
import random
import os

def get_available_xfade_transitions():
    """獲取可用的 xfade 轉場特效"""
    return [
        'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown',
        'slideleft', 'slideright', 'slideup', 'slidedown',
        'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite',
        'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown',
        'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose',
        'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr',
        'coverleft', 'coverright', 'coverup', 'coverdown',
        'revealleft', 'revealright', 'revealup', 'revealdown',
        'squeezeh', 'squeezev', 'hlslice', 'hrslice', 'vuslice', 'vdslice'
    ]

def check_ffmpeg():
    """檢查 FFmpeg 是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def create_video_with_transitions(image_paths, transitions, output_path, 
                                image_duration=3, transition_duration=1, 
                                fps=24, resolution="854:480"):
    """使用指定的轉場特效創建視頻"""
    
    if len(image_paths) < 2:
        raise ValueError("至少需要 2 張圖片")
    
    if len(transitions) != len(image_paths) - 1:
        raise ValueError("轉場數量應該比圖片數量少 1")
    
    total_duration = len(image_paths) * image_duration
    width, height = resolution.split(':')
    
    st.info(f"🎯 預計總時長: {total_duration} 秒")
    
    # 使用所有輸入都有相同長度的方法
    inputs = []
    for img_path in image_paths:
        inputs.extend(['-loop', '1', '-t', str(total_duration), '-i', str(img_path)])
    
    # 構建 filter_complex
    filter_chains = []
    
    # 為每個輸入創建處理鏈
    for i in range(len(image_paths)):
        filter_chains.append(
            f'[{i}:v]fps={fps},'
            f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
            f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,'
            f'setsar=1,'
            f'trim=duration={total_duration},'
            f'setpts=PTS-STARTPTS'
            f'[p{i}]'
        )
    
    # 構建轉場鏈
    xfade_filters = []
    
    # 第一個轉場
    xfade_filters.append(
        f'[p0][p1]xfade=transition={transitions[0]}:'
        f'duration={transition_duration}:'
        f'offset={1*image_duration-transition_duration}'
        f'[x0]'
    )
    
    # 後續轉場
    for i in range(1, len(transitions)):
        xfade_filters.append(
            f'[x{i-1}][p{i+1}]xfade=transition={transitions[i]}:'
            f'duration={transition_duration}:'
            f'offset={(i+1)*image_duration-transition_duration}'
            f'[x{i}]'
        )
    
    # 合併所有 filter
    all_filters = filter_chains + xfade_filters
    filter_complex = ';'.join(all_filters)
    
    # 最終輸出標籤
    final_output = f'x{len(transitions)-1}'
    
    cmd = [
        'ffmpeg', '-y',
    ] + inputs + [
        '-filter_complex', filter_complex,
        '-map', f'[{final_output}]',
        '-t', str(total_duration),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-r', str(fps),
        '-movflags', '+faststart',
        str(output_path)
    ]
    
    st.write("🎬 執行命令:")
    st.info(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0, transitions
    except Exception as e:
        st.error(f"❌ 執行錯誤: {str(e)}")
        return False, []

def get_video_duration(video_path):
    """獲取視頻的實際時長"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.stdout else 0
    except:
        return 0

def get_transition_description(transition):
    """獲取轉場特效的描述"""
    descriptions = {
        'fade': '淡入淡出',
        'wipeleft': '向左擦拭',
        'wiperight': '向右擦拭',
        'wipeup': '向上擦拭',
        'wipedown': '向下擦拭',
        'slideleft': '向左滑動',
        'slideright': '向右滑動',
        'slideup': '向上滑動',
        'slidedown': '向下滑動',
        'fadeblack': '淡入黑色',
        'fadewhite': '淡入白色',
        'circleopen': '圓形展開',
        'circleclose': '圓形閉合',
        'dissolve': '溶解',
        'pixelize': '像素化',
        'radial': '放射狀',
        'coverleft': '向左覆蓋',
        'coverright': '向右覆蓋',
        'coverup': '向上覆蓋',
        'coverdown': '向下覆蓋',
        'revealleft': '向左揭示',
        'revealright': '向右揭示',
        'revealup': '向上揭示',
        'revealdown': '向下揭示',
        'circlecrop': '圓形裁剪',
        'rectcrop': '矩形裁剪',
        'distance': '距離',
        'smoothleft': '平滑向左',
        'smoothright': '平滑向右',
        'smoothup': '平滑向上',
        'smoothdown': '平滑向下',
        'vertopen': '垂直展開',
        'vertclose': '垂直閉合',
        'horzopen': '水平展開',
        'horzclose': '水平閉合'
    }
    return descriptions.get(transition, transition)

def main():
    # 頁面設置
    st.set_page_config(
        page_title="雙模式轉場視頻生成器",
        page_icon="🎬",
        layout="wide"
    )
    
    # 標題和描述
    st.title("🎬 雙模式轉場視頻生成器")
    st.markdown("選擇隨機轉場或為每個轉場指定特效")
    st.markdown("---")
    
    # 檢查 FFmpeg
    if not check_ffmpeg():
        st.error("❌ 請先安裝 FFmpeg")
        return
    
    # 第一步：上傳圖片
    st.header("1. 上傳圖片")
    uploaded_files = st.file_uploader(
        "選擇多張圖片（2張以上）",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("👆 請上傳至少 2 張圖片")
        return
    
    if len(uploaded_files) < 2:
        st.warning("⚠️ 請上傳至少 2 張圖片")
        return
    
    # 顯示圖片預覽
    st.subheader(f"📸 已選擇 {len(uploaded_files)} 張圖片")
    cols = st.columns(min(4, len(uploaded_files)))
    st.info([uploaded_files])
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(uploaded_file, use_container_width=True)
            st.caption(f"圖片 {i+1}")
    
    # 第二步：選擇模式
    st.header("2. 選擇轉場模式")
    
    mode = st.radio(
        "選擇轉場模式",
        ["🎲 隨機轉場模式", "🎯 指定轉場模式"],
        index=0,
        horizontal=True
    )
    
    transitions = get_available_xfade_transitions()
    selected_transitions = []
    
    if "隨機轉場" in mode:
        # 隨機轉場模式
        st.subheader("🎲 隨機轉場設置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 生成隨機轉場序列", use_container_width=True):
                st.session_state.random_transitions = True
        
        with col2:
            if st.button("🔄 重新隨機", use_container_width=True):
                st.session_state.rerandom = True
        
        # 生成隨機轉場序列
        if hasattr(st.session_state, 'random_transitions') or hasattr(st.session_state, 'rerandom'):
            random_transitions = [random.choice(transitions) for _ in range(len(uploaded_files) - 1)]
            selected_transitions = random_transitions
            
            # 顯示隨機生成的轉場序列
            st.subheader("🎲 隨機轉場序列")
            random_cols = st.columns(min(4, len(selected_transitions)))
            for i, transition in enumerate(selected_transitions):
                with random_cols[i % 4]:
                    st.success(f"**轉場 {i+1}**\n`{transition}`\n{get_transition_description(transition)}")
            
            # 清除狀態
            if hasattr(st.session_state, 'random_transitions'):
                del st.session_state.random_transitions
            if hasattr(st.session_state, 'rerandom'):
                del st.session_state.rerandom
        
    else:
        # 指定轉場模式
        st.subheader("🎯 指定轉場設置")
        
        # 為每個轉場點選擇特效
        for i in range(len(uploaded_files) - 1):
            st.markdown(f"### 🎭 轉場 {i+1}: 圖片 {i+1} → 圖片 {i+2}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                transition = st.selectbox(
                    f"選擇轉場特效",
                    transitions,
                    index=i % len(transitions),
                    key=f"transition_{i}",
                    format_func=get_transition_description
                )
                selected_transitions.append(transition)
            
            with col2:
                st.info(f"**{get_transition_description(transition)}**")
        
        # 快速預設按鈕
        st.subheader("🚀 快速預設")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🌅 淡入淡出", use_container_width=True):
                fade_transitions = ['fade', 'fadeblack', 'fadewhite', 'dissolve']
                for i in range(len(uploaded_files) - 1):
                    st.session_state[f"transition_{i}"] = fade_transitions[i % len(fade_transitions)]
                st.rerun()
        
        with col2:
            if st.button("🌀 滑動效果", use_container_width=True):
                slide_transitions = ['slideleft', 'slideright', 'slideup', 'slidedown']
                for i in range(len(uploaded_files) - 1):
                    st.session_state[f"transition_{i}"] = slide_transitions[i % len(slide_transitions)]
                st.rerun()
        
        with col3:
            if st.button("✨ 擦拭效果", use_container_width=True):
                wipe_transitions = ['wipeleft', 'wiperight', 'wipeup', 'wipedown']
                for i in range(len(uploaded_files) - 1):
                    st.session_state[f"transition_{i}"] = wipe_transitions[i % len(wipe_transitions)]
                st.rerun()
        
        with col4:
            if st.button("⭕ 圓形效果", use_container_width=True):
                circle_transitions = ['circleopen', 'circleclose', 'radial', 'circlecrop']
                for i in range(len(uploaded_files) - 1):
                    st.session_state[f"transition_{i}"] = circle_transitions[i % len(circle_transitions)]
                st.rerun()
    
    # 第三步：視頻設置
    st.header("3. 視頻設置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        image_duration = st.slider(
            "每張圖片顯示時間（秒）",
            min_value=2,
            max_value=10,
            value=3
        )
        
        fps = st.selectbox("視頻幀率", [24, 30, 60], index=0)
    
    with col2:
        transition_duration = st.slider(
            "轉場持續時間（秒）",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.5
        )
        
        resolution = st.selectbox(
            "視頻分辨率",
            ["854:480", "1280:720", "1920:1080"],
            index=0
        )
    
    # 驗證設置
    if transition_duration >= image_duration:
        st.error("⚠️ 轉場持續時間必須小於圖片顯示時間")
        return
    
    # 如果還沒有選擇轉場，顯示提示
    if not selected_transitions and "指定轉場" in mode:
        st.warning("⚠️ 請為每個轉場選擇特效")
        return
    
    # 在隨機模式下，如果還沒有生成轉場，自動生成
    if "隨機轉場" in mode and not selected_transitions:
        selected_transitions = [random.choice(transitions) for _ in range(len(uploaded_files) - 1)]
    
    # 顯示轉場預覽
    if selected_transitions:
        st.subheader("🔄 轉場預覽")
        preview_cols = st.columns(min(4, len(selected_transitions)))
        for i, transition in enumerate(selected_transitions):
            with preview_cols[i % 4]:
                if "隨機轉場" in mode:
                    st.success(f"**轉場 {i+1}**\n`{transition}`\n{get_transition_description(transition)}")
                else:
                    st.info(f"**轉場 {i+1}**\n`{transition}`\n{get_transition_description(transition)}")
    
    # 顯示時間線
    total_duration = len(uploaded_files) * image_duration
    st.info(f"**預計視頻總時長**: {total_duration} 秒")
    
    st.subheader("🕒 時間線規劃")
    for i in range(len(uploaded_files)):
        start_time = i * image_duration
        end_time = (i + 1) * image_duration
        st.write(f"🖼️ 圖片 {i+1}: {start_time}-{end_time}秒")
        
        if i < len(uploaded_files) - 1 and selected_transitions:
            transition_start = end_time - transition_duration
            transition_name = get_transition_description(selected_transitions[i])
            st.write(f"   🎭 轉場 {i+1} ({transition_name}): {transition_start}-{end_time}秒")
    
    # 第四步：生成視頻
    st.header("4. 生成視頻")
    
    if st.button("🎥 生成視頻", type="primary", use_container_width=True):
        if not selected_transitions:
            st.error("❌ 請先選擇轉場模式並設置轉場特效")
            return
            
        with st.spinner("正在生成視頻，請稍候..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 保存上傳的圖片
                image_paths = []
                for i, uploaded_file in enumerate(uploaded_files):
                    image_path = temp_path / f"image_{i:03d}.{uploaded_file.name.split('.')[-1]}"
                    with open(image_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    image_paths.append(image_path)
                
                # 生成視頻
                output_path = temp_path / "output_video.mp4"
                success, used_transitions = create_video_with_transitions(
                    image_paths, selected_transitions, output_path, 
                    image_duration, transition_duration, fps, resolution
                )
                
                if success and output_path.exists():
                    # 驗證最終時長
                    actual_duration = get_video_duration(output_path)
                    
                    if abs(actual_duration - total_duration) <= 0.5:
                        st.success(f"✅ 視頻生成成功！時長: {actual_duration:.2f} 秒")
                    else:
                        st.warning(f"⚠️ 視頻生成完成，但時長有偏差: 預計 {total_duration}秒, 實際 {actual_duration:.2f}秒")
                    
                    # 顯示轉場信息
                    st.subheader("🎭 使用的轉場特效")
                    trans_cols = st.columns(min(4, len(used_transitions)))
                    for i, transition in enumerate(used_transitions):
                        with trans_cols[i % 4]:
                            if "隨機轉場" in mode:
                                st.success(f"**轉場 {i+1}**\n`{transition}`\n{get_transition_description(transition)}")
                            else:
                                st.info(f"**轉場 {i+1}**\n`{transition}`\n{get_transition_description(transition)}")
                    
                    # 顯示視頻
                    st.subheader("📺 視頻預覽")
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    # 視頻信息
                    video_size = os.path.getsize(output_path) / (1024 * 1024)
                    st.info(f"""
                    **📊 視頻信息：**
                    - 模式: {mode}
                    - 預計時長: {total_duration} 秒
                    - 實際時長: {actual_duration:.2f} 秒
                    - 文件大小: {video_size:.1f} MB
                    - 分辨率: {resolution.replace(':', 'x')}
                    - 幀率: {fps} FPS
                    - 轉場數量: {len(used_transitions)}
                    - 圖片數量: {len(uploaded_files)}
                    """)
                    
                    # 下載按鈕
                    file_name = "random_transition_video.mp4" if "隨機轉場" in mode else "custom_transition_video.mp4"
                    st.download_button(
                        label="📥 下載視頻",
                        data=video_bytes,
                        file_name=file_name,
                        mime="video/mp4",
                        use_container_width=True
                    )
                    
                else:
                    st.error("❌ 視頻生成失敗，請調整設置後重試")

if __name__ == "__main__":
    main()
