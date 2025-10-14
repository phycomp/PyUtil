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
        'revealleft', 'revealright', 'revealup', 'revealdown'
    ]

def check_ffmpeg():
    """檢查 FFmpeg 是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def create_video_with_concat_method(image_paths, output_path, 
                                  image_duration=3, transition_duration=1, 
                                  fps=24, resolution="854:480"):
    """使用 concat 方法確保正確時長"""
    
    if len(image_paths) < 2:
        raise ValueError("至少需要 2 張圖片")
    
    transitions = get_available_xfade_transitions()
    random_transitions = [random.choice(transitions) for _ in range(len(image_paths) - 1)]
    
    total_duration = len(image_paths) * image_duration
    width, height = resolution.split(':')
    
    st.info(f"🎯 預計總時長: {total_duration} 秒")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 方法：使用單一 filter_complex 處理所有輸入，確保正確時長控制
        inputs = []
        for i, img_path in enumerate(image_paths):
            # 使用 -loop 1 但配合正確的時長控制
            inputs.extend(['-loop', '1', '-t', str(image_duration + transition_duration), '-i', str(img_path)])
        
        # 構建 filter_complex - 關鍵修正
        filter_parts = []
        
        # 1. 處理所有輸入流，確保每個都有正確的時長
        for i in range(len(image_paths)):
            # 使用 setpts 重置時間戳，trim 確保時長
            filter_parts.append(
                f'[{i}:v]fps={fps},'
                f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,'
                f'setsar=1,'
                f'setpts=PTS-STARTPTS,'  # 重置時間戳
                f'trim=duration={image_duration + transition_duration}'  # 確保時長
                f'[v{i}]'
            )
        
        # 2. 構建轉場鏈 - 使用正確的 offset
        current = 'v0'
        for i in range(len(random_transitions)):
            # offset 計算：每個轉場在前一個片段結束時開始
            offset = i * image_duration
            
            filter_parts.append(
                f'[{current}][v{i+1}]'
                f'xfade=transition={random_transitions[i]}:'
                f'duration={transition_duration}:'
                f'offset={offset}[x{i}]'
            )
            current = f'x{i}'
        
        filter_complex = ';'.join(filter_parts)
        
        cmd = [
            'ffmpeg', '-y',
        ] + inputs + [
            '-filter_complex', filter_complex,
            '-map', f'[{current}]',
            '-t', str(total_duration),  # 總時長控制
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-r', str(fps),
            '-movflags', '+faststart',
            output_path
        ]
        
        st.write("🎬 執行命令:")
        st.code(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return result.returncode == 0, random_transitions
        except Exception as e:
            st.error(f"❌ 執行錯誤: {str(e)}")
            return False, []

def create_video_working_method(image_paths, output_path, 
                              image_duration=3, transition_duration=1, 
                              fps=24, resolution="854:480"):
    """經過測試的可靠方法"""
    
    transitions = get_available_xfade_transitions()
    random_transitions = [random.choice(transitions) for _ in range(len(image_paths) - 1)]
    
    total_duration = len(image_paths) * image_duration
    width, height = resolution.split(':')
    
    st.info(f"🎯 預計總時長: {total_duration} 秒")
    
    # 直接使用單一 filter_complex，正確處理時長
    inputs = []
    for i, img_path in enumerate(image_paths):
        # 關鍵：每個輸入的時長要足夠長，覆蓋整個時間線
        inputs.extend(['-loop', '1', '-t', str(total_duration), '-i', str(img_path)])
    
    # 構建 filter_complex
    filter_parts = []
    
    # 處理所有輸入流
    for i in range(len(image_paths)):
        filter_parts.append(
            f'[{i}:v]fps={fps},'
            f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
            f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,'
            f'setsar=1,'
            f'trim=duration={total_duration},'  # 確保足夠的時長
            f'setpts=PTS-STARTPTS'  # 重置時間戳
            f'[v{i}]'
        )
    
    # 構建轉場鏈 - 關鍵修正：offset 從 0 開始累加
    current = 'v0'
    for i in range(len(random_transitions)):
        # 正確的 offset 計算：每個轉場在前一個片段顯示時間結束時開始
        offset = (i + 1) * image_duration - transition_duration
        
        filter_parts.append(
            f'[{current}][v{i+1}]'
            f'xfade=transition={random_transitions[i]}:'
            f'duration={transition_duration}:'
            f'offset={offset}[x{i}]'
        )
        current = f'x{i}'
    
    filter_complex = ';'.join(filter_parts)
    
    cmd = [
        'ffmpeg', '-y',
    ] + inputs + [
        '-filter_complex', filter_complex,
        '-map', f'[{current}]',
        '-t', str(total_duration),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-r', str(fps),
        '-movflags', '+faststart',
        output_path
    ]
    
    st.write("🎬 執行命令:")
    st.code(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0, random_transitions
    except Exception as e:
        st.error(f"❌ 執行錯誤: {str(e)}")
        return False, []

def create_video_simple_reliable(image_paths, output_path, 
                               image_duration=3, transition_duration=1, 
                               fps=24, resolution="854:480"):
    """最簡單可靠的方法"""
    
    transitions = get_available_xfade_transitions()
    random_transitions = [random.choice(transitions) for _ in range(len(image_paths) - 1)]
    
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
        f'[p0][p1]xfade=transition={random_transitions[0]}:'
        f'duration={transition_duration}:'
        f'offset={1*image_duration-transition_duration}'
        f'[x0]'
    )
    
    # 後續轉場
    for i in range(1, len(random_transitions)):
        xfade_filters.append(
            f'[x{i-1}][p{i+1}]xfade=transition={random_transitions[i]}:'
            f'duration={transition_duration}:'
            f'offset={(i+1)*image_duration-transition_duration}'
            f'[x{i}]'
        )
    
    # 合併所有 filter
    all_filters = filter_chains + xfade_filters
    filter_complex = ';'.join(all_filters)
    
    # 最終輸出標籤
    final_output = f'x{len(random_transitions)-1}'
    
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
        return result.returncode == 0, random_transitions
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

def main():
    # 頁面設置
    st.set_page_config(
        page_title="隨機轉場視頻生成器",
        page_icon="🎬",
        layout="centered"
    )
    
    # 標題和描述
    st.title("🎬 隨機轉場視頻生成器")
    st.markdown("上傳圖片 → 自動添加酷炫轉場 → 下載視頻")
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
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(uploaded_file, use_container_width=True)
            st.caption(f"圖片 {i+1}")
    
    # 第二步：設置
    st.header("2. 視頻設置")
    
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
    
    # 顯示預計時長
    total_duration = len(uploaded_files) * image_duration
    st.info(f"**預計視頻總時長**: {total_duration} 秒")
    
    # 第三步：生成視頻
    st.header("3. 生成視頻")
    
    if st.button("🎥 一鍵生成視頻", type="primary", use_container_width=True):
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
                
                # 生成視頻 - 使用最可靠的方法
                output_path = temp_path / "output_video.mp4"
                success, used_transitions = create_video_simple_reliable(
                    image_paths, output_path, image_duration, transition_duration, fps, resolution
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
                    for i, transition in enumerate(used_transitions):
                        st.write(f"**轉場 {i+1}**: `{transition}`")
                    
                    # 顯示視頻
                    st.subheader("📺 視頻預覽")
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    # 下載按鈕
                    st.download_button(
                        label="📥 下載視頻",
                        data=video_bytes,
                        file_name="random_transition_video.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    
                else:
                    st.error("❌ 視頻生成失敗，請調整設置後重試")

if __name__ == "__main__":
    main()
